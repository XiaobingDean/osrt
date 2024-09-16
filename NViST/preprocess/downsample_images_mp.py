import os
from PIL import Image
import glob
import argparse
from queue import Queue  # Import for queue
import tqdm
import threading

# Function to open, resize, and save an image
def resize_and_save_image(imgfile, savedir_12):
    fname = imgfile.split('/')[-1]
    #print(savedir_12, fname, imgfile.split('/')[-1], imgfile.split('/'))
    savepath_12 = os.path.join(savedir_12, fname)
    if not os.path.isfile(savepath_12):
        try:
            with Image.open(imgfile) as img:
                width, height = img.size
                if width > height : 
                    scale = 160 / width
                    print(imgfile)
                else:
                    scale = 160 / height
                    new_width, new_height = int(width * scale), int(height * scale)
                    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                    print(resized_img)
                    resized_img.save(savepath_12)
        except OSError as e:
            if 'image file is truncated' in str(e):
                print(f'Error loading image: {savepath_12}. The image file may be truncated or corrupted.')

def main():
    parser = argparse.ArgumentParser(description='Downsample images')
    parser.add_argument("-d", "--data_dir", type=str, default="../../data/mvimgnet_original", help="where the dataset is")
    # parser.add_argument("-r", "--multi_resolution", type=str2bool, default=False, help="if yes, we also downsample images by 3 and 6")
    parser.add_argument("-r", "--multi_resolution", default=False, help="if yes, we also downsample images by 3 and 6")
    parser.add_argument("-t", "--threads", default=64, help="Number of threads used")
    args = parser.parse_args()
    
    image_queue = Queue()

    # ... (loop through directories and scenes)
    dirnames = os.listdir(args.data_dir)[::-1]
    dirnames = [dirname for dirname in dirnames if os.path.isdir(os.path.join(args.data_dir, dirname))]

    for dirname in dirnames:
        scenenames = sorted(os.listdir(os.path.join(args.data_dir, dirname)))
        cat_name = {}
        for scenename in scenenames:
            imgfiles = sorted(glob.glob(os.path.join(args.data_dir,dirname,scenename,'images/*.jpg')))
            savedir_12 = os.path.join(args.data_dir,dirname,scenename,'images_12')
            os.makedirs(savedir_12, exist_ok=True)
            
            scale = 0.0
            if len(imgfiles) > 0:
                #print(dirname, scenename, savedir_12, scale)
                for imgfile in imgfiles:
                    # if imgfile.find('../data/1/25000041/images') != -1:
                    #     print(imgfile)
                    if imgfile.find('../data/1/25000041/images') != -1 and imgfile.endswith(".jpg") or imgfile.endswith(".png"):
                        # print(imgfile, savedir_12)
                        image_queue.put([imgfile, savedir_12])

    # ... (create thread pool)

    # Start worker threads
    pbar = tqdm.tqdm(total = image_queue.qsize())
    for _ in range(args.threads):  # Replace with actual number of threads
        thread = threading.Thread(target=worker, args=(image_queue, pbar))
        thread.start()

    image_queue.join()

    print('Downsample finished.')

def worker(image_queue, pbar):
    while not image_queue.empty():
        list_ = image_queue.get()
        # ... (logic to determine save path based on multi_resolution flag)
        resize_and_save_image(list_[0], list_[1])
        image_queue.task_done()  # Signal task completion
        pbar.update(1)

if __name__ == "__main__":
    main()