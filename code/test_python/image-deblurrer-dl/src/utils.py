import torch  

def find_max_batch_size(model, dataset, start_batch=32):
    batch_size = start_batch
    while batch_size > 0:
        try:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)
            sample = next(iter(loader))  
            with torch.no_grad():
                model(sample[0].to("cuda"))
            print(f"✔️ Optimal batch size: {batch_size}")
            return batch_size
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                batch_size = batch_size // 2  
            else:
                raise e

    print("❌ No valid batch size found")
    return 1  
