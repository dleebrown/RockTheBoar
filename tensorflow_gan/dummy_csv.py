import os

test_image_dir = '/home/donald/Desktop/PYTHON/kaggle_car_competition/train/'
dummy_csv = '/home/donald/Desktop/test_entry.csv'


def get_list_of_images(image_dir):
    image_list = [img for img in os.listdir(image_dir) if '.jpg' in img]
    return image_list


def write_dummy_entry(image_name, open_csv):
    rle_string = '1 1'
    open_csv.write(image_name+','+rle_string+'\n')

if __name__ == '__main__':

    file_output = open(dummy_csv, mode='w')
    file_output.write('img,rle_mask\n')
    images = get_list_of_images(test_image_dir)
    for i in range(len(images)):
        write_dummy_entry(images[i], file_output)

    file_output.close()
