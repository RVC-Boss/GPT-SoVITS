def register_Hanlder(app):
  from handlers import index_router
  app.include_router(index_router)
