"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import cv2

# try:
#     import wandb
# except ImportError:
#     print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

class cyclegan():
    def __init__(self):
        self.opt=0
        self.dataset=0
        self.model=0


    def setup_model(self):
        self.opt = TestOptions().parse()  # get test options
        # hard-code some parameters for test
        self.opt.num_threads = 0   # test code only supports num_threads = 0
        self.opt.batch_size = 1    # test code only supports batch_size = 1
        self.opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        self.opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
        self.dataset = create_dataset(self.opt)  # create a dataset given opt.dataset_mode and other options
        self.model = create_model(self.opt)      # create a model given opt.model and other options
        self.model.setup(self.opt)               # regular setup: load and print networks; create schedulers

        # initialize logger
        # if opt.use_wandb:
        #     wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        #     wandb_run._label(repo='CycleGAN-and-pix2pix')

        # create a website
        # web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
        # if opt.load_iter > 0:  # load_iter is 0 by default
        #     web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
        # print('creating web directory', web_dir)
        # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        if self.opt.eval:
            self.model.eval()


    def get_pic(self):
        # web_dir = os.path.join(self.opt.results_dir, self.opt.name, '{}_{}'.format(self.opt.phase, self.opt.epoch))  # define the website directory
        # if self.opt.load_iter > 0:  # load_iter is 0 by default
        #     web_dir = '{:s}_iter{:d}'.format(web_dir, self.opt.load_iter)
        # print('creating web directory', web_dir)
        # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (self.opt.name, self.opt.phase, self.opt.epoch))
        for i, data in enumerate(self.dataset):
            if i >= self.opt.num_test:  # only apply our model to opt.num_test images.
                break
            # print(data)
            self.model.set_input(data)  # unpack data from data loader
            self.model.test()   
            visuals = self.model.get_current_visuals()  # get image results
            img_path = self.model.get_image_paths()     # get image paths
            frame = save_images(visuals, img_path, aspect_ratio=self.opt.aspect_ratio, width=self.opt.display_winsize, use_wandb=self.opt.use_wandb)


        cv2.imwrite('C:/Users/member/Desktop/pytorch-CycleGAN-and-pix2pix-master/datasets/simtoreal1/sim2.jpg',frame)

        os.remove('datasets/simtoreal1/sim.jpg')

def main():
    gan=cyclegan()
    gan.setup_model()
    capture1 = cv2.VideoCapture(0)
    capture1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    capture1.set(cv2.CAP_PROP_FPS, 60)           # カメラFPSを60FPSに設定
    capture1.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # カメラ画像の横幅を1280に設定
    capture1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # カメラ画像の縦幅を720に設定
    while(1):
        ret1, frame1 = capture1.read()
        cv2.imwrite('C:/Users/member/Desktop/pytorch-CycleGAN-and-pix2pix-master/datasets/simtoreal1/sim.jpg',frame1)
    
        gan.get_pic()
        print("a")

if __name__ == '__main__':
    main()



        
