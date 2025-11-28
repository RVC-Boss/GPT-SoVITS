import torch
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.distributed import _find_tensors
from packaging import version


# from https://github.com/Lightning-AI/lightning-bolts/blob/5d61197cd2f491f69e238137a5edabe80ae14ad9/pl_bolts/models/self_supervised/simclr/simclr_module.py#L20
class SyncFunction(torch.autograd.Function):
    @staticmethod
    # @torch.no_grad()
    def forward(ctx, tensor):
        world_size = torch.distributed.get_world_size()

        # Collect batch sizes from all processes
        local_bs = torch.tensor([tensor.shape[0]], device=tensor.device)
        batch_sizes = [torch.zeros_like(local_bs) for _ in range(world_size)]
        torch.distributed.all_gather(batch_sizes, local_bs)

        # Convert to integer list and find the minimum
        batch_sizes_int = [bs.item() for bs in batch_sizes]
        min_bs = min(batch_sizes_int)

        # Crop the tensor to the minimum batch size if needed
        cropped_tensor = tensor[:min_bs] if tensor.shape[0] > min_bs else tensor

        # Prepare for gathering
        out_shape = (min_bs * world_size,) + tensor.shape[1:]
        gathered_tensor = torch.zeros(out_shape, dtype=tensor.dtype, device=tensor.device)

        # Build tensor list for all_gather
        tensor_list = list(torch.chunk(gathered_tensor, world_size))

        # Perform all_gather using the cropped tensors
        torch.distributed.all_gather(tensor_list, cropped_tensor)

        # Save for backward pass
        ctx.min_bs = min_bs
        ctx.world_size = world_size
        ctx.orig_shape = tensor.shape

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        assert False
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]

class DDP(DistributedDataParallel):
    """
    Override the forward call in lightning so it goes to training and validation step respectively
    """

    def forward(self, *inputs, **kwargs):  # pragma: no cover
        if version.parse(torch.__version__[:6]) < version.parse("1.11"):
            self._sync_params()
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            assert len(self.device_ids) == 1
            if self.module.training:
                output = self.module.training_step(*inputs[0], **kwargs[0])
            elif self.module.testing:
                output = self.module.test_step(*inputs[0], **kwargs[0])
            else:
                output = self.module.validation_step(*inputs[0], **kwargs[0])
            if torch.is_grad_enabled():
                # We'll return the output object verbatim since it is a freeform
                # object. We need to find any tensors in this object, though,
                # because we need to figure out which parameters were used during
                # this forward pass, to ensure we short circuit reduction for any
                # unused parameters. Only if `find_unused_parameters` is set.
                if self.find_unused_parameters:
                    self.reducer.prepare_for_backward(list(_find_tensors(output)))
                else:
                    self.reducer.prepare_for_backward([])
        else:
            from torch.nn.parallel.distributed import (
                Join,
                _DDPSink,
                _tree_flatten_with_rref,
                _tree_unflatten_with_rref,
            )

            with torch.autograd.profiler.record_function("DistributedDataParallel.forward"):
                if torch.is_grad_enabled() and self.require_backward_grad_sync:
                    self.logger.set_runtime_stats_and_log()
                    self.num_iterations += 1
                    self.reducer.prepare_for_forward()

                # Notify the join context that this process has not joined, if
                # needed
                work = Join.notify_join_context(self)
                if work:
                    self.reducer._set_forward_pass_work_handle(work, self._divide_by_initial_world_size)

                # Calling _rebuild_buckets before forward compuation,
                # It may allocate new buckets before deallocating old buckets
                # inside _rebuild_buckets. To save peak memory usage,
                # call _rebuild_buckets before the peak memory usage increases
                # during forward computation.
                # This should be called only once during whole training period.
                if torch.is_grad_enabled() and self.reducer._rebuild_buckets():
                    print("Reducer buckets have been rebuilt in this iteration.")
                    self._has_rebuilt_buckets = True

                # sync params according to location (before/after forward) user
                # specified as part of hook, if hook was specified.
                buffer_hook_registered = hasattr(self, "buffer_hook")
                if self._check_sync_bufs_pre_fwd():
                    self._sync_buffers()

                if self._join_config.enable:
                    # Notify joined ranks whether they should sync in backwards pass or not.
                    self._check_global_requires_backward_grad_sync(is_joined_rank=False)

                inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
                if self.module.training:
                    output = self.module.training_step(*inputs[0], **kwargs[0])
                elif self.module.testing:
                    output = self.module.test_step(*inputs[0], **kwargs[0])
                else:
                    output = self.module.validation_step(*inputs[0], **kwargs[0])

                # sync params according to location (before/after forward) user
                # specified as part of hook, if hook was specified.
                if self._check_sync_bufs_post_fwd():
                    self._sync_buffers()

                if torch.is_grad_enabled() and self.require_backward_grad_sync:
                    self.require_forward_param_sync = True
                    # We'll return the output object verbatim since it is a freeform
                    # object. We need to find any tensors in this object, though,
                    # because we need to figure out which parameters were used during
                    # this forward pass, to ensure we short circuit reduction for any
                    # unused parameters. Only if `find_unused_parameters` is set.
                    if self.find_unused_parameters and not self.static_graph:
                        # Do not need to populate this for static graph.
                        self.reducer.prepare_for_backward(list(_find_tensors(output)))
                    else:
                        self.reducer.prepare_for_backward([])
                else:
                    self.require_forward_param_sync = False

            # TODO: DDPSink is currently enabled for unused parameter detection and
            # static graph training for first iteration.
            if (self.find_unused_parameters and not self.static_graph) or (
                self.static_graph and self.num_iterations == 1
            ):
                state_dict = {
                    "static_graph": self.static_graph,
                    "num_iterations": self.num_iterations,
                }

                output_tensor_list, treespec, output_is_rref = _tree_flatten_with_rref(output)
                output_placeholders = [None for _ in range(len(output_tensor_list))]
                # Do not touch tensors that have no grad_fn, which can cause issues
                # such as https://github.com/pytorch/pytorch/issues/60733
                for i, output in enumerate(output_tensor_list):
                    if torch.is_tensor(output) and output.grad_fn is None:
                        output_placeholders[i] = output

                # When find_unused_parameters=True, makes tensors which require grad
                # run through the DDPSink backward pass. When not all outputs are
                # used in loss, this makes those corresponding tensors receive
                # undefined gradient which the reducer then handles to ensure
                # param.grad field is not touched and we don't error out.
                passthrough_tensor_list = _DDPSink.apply(
                    self.reducer,
                    state_dict,
                    *output_tensor_list,
                )
                for i in range(len(output_placeholders)):
                    if output_placeholders[i] is None:
                        output_placeholders[i] = passthrough_tensor_list[i]

                # Reconstruct output data structure.
                output = _tree_unflatten_with_rref(output_placeholders, treespec, output_is_rref)
        return output
