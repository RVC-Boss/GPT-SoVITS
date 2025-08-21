import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import numpy as np
import os

# Define file paths
PLAYGROUND_DIR = "playground"
MODEL_A_PATH = os.path.join(PLAYGROUND_DIR, "a.onnx")
MODEL_B_PATH = os.path.join(PLAYGROUND_DIR, "b.onnx")
MODEL_C_PATH = os.path.join(PLAYGROUND_DIR, "c.onnx")

# --- 1. Create two simple PyTorch modules ---

class ModelA(nn.Module):
    """This model adds 1 to the input."""
    def forward(self, x):
        return x + 1.0

class ModelB(nn.Module):
    """This model multiplies the input by 2."""
    def forward(self, x):
        return x * 2.0

def create_and_export_models():
    """Creates two nn.Modules and exports them to ONNX."""
    print("Step 1: Creating and exporting PyTorch models A and B...")
    os.makedirs(PLAYGROUND_DIR, exist_ok=True)

    # Define a dummy input with a dynamic axis
    batch_size = 1
    sequence_length = 10  # This dimension will be dynamic
    features = 4
    dummy_input = torch.randn(batch_size, sequence_length, features)

    # Export Model A
    print(f"Exporting Model A to {MODEL_A_PATH}")
    torch.onnx.export(
        ModelA(),
        dummy_input,
        MODEL_A_PATH,
        input_names=['inputA'],
        output_names=['output'],
        dynamic_axes={'inputA': {1: 'sequenceA'}, 'output': {1: 'sequence'}},
        opset_version=11 # If node requires opset >= 11
    )

    # Export Model B
    print(f"Exporting Model B to {MODEL_B_PATH}")
    torch.onnx.export(
        ModelB(),
        dummy_input,
        MODEL_B_PATH,
        input_names=['inputB'],
        output_names=['output'],
        dynamic_axes={'inputB': {1: 'sequenceB'}, 'output': {1: 'sequence'}},
        opset_version=11
    )
    print("Models A and B exported successfully.")

def combine_models_with_if():
    """
    Reads two ONNX models and combines them into a third model
    using an 'If' operator.
    """
    print("\nStep 2: Combining models A and B into C with an 'If' node...")

    # Load the two exported ONNX models
    model_a = onnx.load(MODEL_A_PATH)
    model_b = onnx.load(MODEL_B_PATH)

    # The graphs for the 'then' and 'else' branches of the 'If' operator
    then_graph = model_a.graph
    then_graph.name = "then_branch_graph"
    else_graph = model_b.graph
    else_graph.name = "else_branch_graph"

    # The data input for the main graph is defined here.
    # We take it from one of the original models.
    data_inputA = model_a.graph.input[0]
    data_inputB = model_b.graph.input[0]

    # For some onnxruntime versions, subgraphs should not have their own
    # explicit 'input' list if the inputs are captured from the parent graph.
    # We clear the input lists of the subgraphs to force implicit capture.
    del then_graph.input[:]
    del else_graph.input[:]

    # The output names of the subgraphs must be the same.
    # The 'If' node will have an output with this same name.
    subgraph_output_name = model_a.graph.output[0].name
    assert subgraph_output_name == model_b.graph.output[0].name, "Subgraph output names must match"


    # Define the inputs for the main graph
    # 1. The boolean condition to select the branch
    cond_input = helper.make_tensor_value_info('if_use_a', TensorProto.BOOL, [])

    # The main graph's output is the output from the 'If' node.
    # We can use the ValueInfoProto from one of the subgraphs directly.
    main_output = model_a.graph.output[0]

    # Create the 'If' node
    if_node = helper.make_node(
        'If',
        inputs=['if_use_a'],
        outputs=[subgraph_output_name], # This name MUST match the subgraph's output name
        then_branch=then_graph,
        else_branch=else_graph
    )

    # Create the main graph containing the 'If' node. Its inputs are the condition
    # AND the data that the subgraphs will capture.
    main_graph = helper.make_graph(
        nodes=[if_node],
        name='if_main_graph',
        inputs=[cond_input, data_inputA, data_inputB],
        outputs=[main_output]
    )

    # Create the final combined model, specifying the opset and IR version
    opset_version = 16
    final_model = helper.make_model(main_graph,
                                    producer_name='onnx-if-combiner',
                                    ir_version=9,  # For compatibility with older onnxruntime
                                    opset_imports=[helper.make_opsetid("", opset_version)])

    # Check the model for correctness
    onnx.checker.check_model(final_model)

    # Save the combined model
    onnx.save(final_model, MODEL_C_PATH)
    print(f"Combined model C saved to {MODEL_C_PATH}")

def verify_combined_model():
    """
    Loads the combined ONNX model and runs inference to verify
    that the 'If' branching and dynamic shapes work correctly.
    """
    print("\nStep 3: Verifying the combined model C...")
    sess = ort.InferenceSession(MODEL_C_PATH)

    # --- Test Case 1: Select Model A (if_use_a = True) ---
    print("\n--- Verifying 'then' branch (Model A) ---")
    use_a = np.array(True)
    # Use a different sequence length to test dynamic axis
    test_seq_len_a = 15
    test_seq_len_b = 10
    input_data_a = np.random.randn(1, test_seq_len_a, 4).astype(np.float32)
    input_data_b = np.random.randn(1, test_seq_len_a, 4).astype(np.float32)

    # Run inference
    outputs = sess.run(
        None,
        {'if_use_a': use_a, 'inputA': input_data_a, 'inputB': input_data_b}
    )
    result_a = outputs[0]

    # Calculate expected output from Model A
    expected_a = input_data_a + 1.0

    # Verify the output and shape
    np.testing.assert_allclose(result_a, expected_a, rtol=1e-5, atol=1e-5)
    assert result_a.shape[1] == test_seq_len_a, "Dynamic shape failed for branch A"
    print("✅ Branch A (if_use_a=True) works correctly.")
    print(f"✅ Dynamic shape test passed (input seq_len={test_seq_len_a}, output seq_len={result_a.shape[1]})")

    # --- Test Case 2: Select Model B (if_use_a = False) ---
    print("\n--- Verifying 'else' branch (Model B) ---")
    use_b = np.array(False)
    # Use another sequence length
    test_seq_len_a = 8
    test_seq_len_b = 5
    input_data_a = np.random.randn(1, test_seq_len_a, 4).astype(np.float32)
    input_data_b = np.random.randn(1, test_seq_len_b, 4).astype(np.float32)

    # Run inference
    outputs = sess.run(
        None,
        {'if_use_a': use_b, 'inputA': input_data_a, 'inputB': input_data_b}
    )
    result_b = outputs[0]

    # Calculate expected output from Model B
    expected_b = input_data_b * 2.0

    # Verify the output and shape
    np.testing.assert_allclose(result_b, expected_b, rtol=1e-5, atol=1e-5)
    assert result_b.shape[1] == test_seq_len_b, "Dynamic shape failed for branch B"
    print("✅ Branch B (if_use_a=False) works correctly.")
    print(f"✅ Dynamic shape test passed (input seq_len={test_seq_len_b}, output seq_len={result_b.shape[1]})")

def cleanup():
    """Removes the intermediate ONNX files."""
    print("\nCleaning up intermediate files...")
    for path in [MODEL_A_PATH, MODEL_B_PATH]:
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed {path}")

def main():
    """Main function to run the entire process."""
    try:
        create_and_export_models()
        combine_models_with_if()
        verify_combined_model()
    finally:
        cleanup()
    print("\nAll steps completed successfully!")

if __name__ == "__main__":
    main()
