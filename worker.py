def worker_init_fn(worker_id):
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    