testset_root = '/data0/donghyeon/ipnn/AIM2020VTSRC/test'   # put your testing input folder here
test_size = (1280, 720)          # speficy the frame resolution (640, 360) 
test_crop_size = (1280, 720)

mean = [0.0, 0.0, 0.0]
std  = [1, 1, 1]

inter_frames = 3     # number of interpolated frames
preserve_input = True  # whether to preserve the input frames in the store path


model = 'AcSloMoS_scope_unet_residual_synthesis_edge_LSE'  
pwc_path = './utils/network-default.pytorch'


store_path = 'outputs/running_test'          # where to store the outputs
# checkpoint = 'checkpoints/Stage123_scratch/Stage123_scratch_checkpoint.ckpt'
checkpoint = 'checkpoints/Reproduce001_EQVI_from_scratch_5Lap_10L1/AcSloMo200.ckpt'
