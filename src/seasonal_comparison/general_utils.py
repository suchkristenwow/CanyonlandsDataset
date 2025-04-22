import numpy as np

def robust_load_csv(path, min_cols=4, skip_header=1):
    data = np.genfromtxt(path,skip_header=skip_header) 

    try:
        if not np.isnan(data).all():
            print(f"✅ Loaded annotations using no delimiter")
            return data
    except:
        print("path:",path)
        print(data)

    delimiters = [',', '\t', ';']
    for delim in delimiters:
        try:
            data = np.genfromtxt(path, delimiter=delim, skip_header=skip_header)
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)

            if not np.isnan(data).all() and data.shape[1] >= min_cols:
                #print(f"✅ Loaded annotations using delimiter '{delim}'")
                return data
        except Exception as e:
            print(f"⚠️ Failed to load with delimiter '{delim}': {e}")

    raise ValueError(f"❌ Failed to load usable data from {path} with common delimiters.")
