import mesmerize_core as mc

def run_mesmerize(df):
    for i, row in df.iterrows():
        if row["outputs"] is not None: # item has already been run
            print(f"Skipping row {i} because it has already been run")
            continue # skip
            
        process = row.caiman.run()
        
        # on Windows you MUST reload the batch dataframe after every iteration because it uses the `local` backend.
        # this is unnecessary on Linux & Mac
        # "DummyProcess" is used for local backend so this is automatic
        if process.__class__.__name__ == "DummyProcess":
            df = df.caiman.reload_from_disk()

def create_batch(p_root):
    # set up mesmerize
    mc.set_parent_raw_data_path(p_root)
    batch_path = mc.get_parent_raw_data_path() / "mesmerize-batch/batch.pickle"

    if batch_path.is_file():
        batch_path.unlink()

    # create a new batch
    df = mc.create_batch(batch_path)
    
    return df

def load_batch(p_root):
    # set up mesmerize
    mc.set_parent_raw_data_path(p_root)
    batch_path = mc.get_parent_raw_data_path() / "mesmerize-batch/batch.pickle"

    # to load existing batches use `load_batch()`
    df = mc.load_batch(batch_path)
    
    return df