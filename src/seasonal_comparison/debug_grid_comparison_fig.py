import pickle 
from image_stitching_utils import plot_stitched_summary_grid

output_path = "/media/kristen/easystore2/RestorebotData/matches_fused/fused_1650825680195219039.png"

with open("./may_pano_dict.pickle","rb") as f:
    may_pano_dict = pickle.load(f)
        
with open("./nov_pano_dict.pickle","rb") as f:
    nov_pano_dict = pickle.load(f)

plot_stitched_summary_grid(may_pano_dict,nov_pano_dict,output_path)
