import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image
from os import walk

from classification import classification
from constants import IMAGES, ORIGINAL, NOISED
from denoise import denoise_image
from utils import noise, display


def menu(lst: list) -> str:
    """Function prints menu, awaits selection and returns command"""

    # Print all menu items with numbers 1..n
    for z, y in enumerate(lst):
        print(z + 1, ':', y)
    # Create list with menu items
    nums = [str(z + 1) for z in range(len(lst))]
    nums_str = ', '.join(nums)
    # receive correct command
    while True:
        command = input(f'Enter command ({nums_str}): ')
        if command in nums:
            break
        else:
            print('Wrong command.')
    return command


def define_file() -> str:
    """Function read files in 'images' folder, and asks user to chose file"""

    print('===SELECT FILE===')
    filenames = next(walk(IMAGES), (None, None, []))[2]
    filenames.append('Exit the program')
    command = menu(filenames)
    if int(command) == len(filenames):
        exit()
    else:
        return filenames[int(command) - 1]


def main() -> None:
    """Main function operates program"""

    # Define file name
    file_name = define_file()

    # Open chosen image
    img = image.load_img(IMAGES + file_name, color_mode='grayscale')

    # Show user image
    plt.imshow(img)
    plt.show()

    # Convert image to array
    img_arr = image.img_to_array(img)
    img_arr = np.array([img_arr])
    img_arr = img_arr.astype("float32") / 255.0

    # Classify image
    img_class = classification(img_arr)

    # Suggest user add noise to original image
    if img_class == ORIGINAL:
        while True:
            command = input('Seems like your image is original. Do you want to add noise? y/n: ')
            if command.strip().lower() in ('y', 'yes'):
                noisy_array = noise(img_arr)
                display(img_arr, noisy_array)
                img = image.array_to_img(noisy_array[0])
                img.save(IMAGES + file_name[:-4] + '_noise' + file_name[-4:])
                main()
            elif command.strip().lower() in ('n', 'no'):
                main()
            else:
                continue

    # Suggest user remove noise from noised image
    elif img_class == NOISED:
        while True:
            command = input('Seems like your image has noise. Do you want to remove noise? y/n: ')
            if command.strip().lower() in ('y', 'yes'):
                denoise_array = denoise_image(img_arr)
                display(img_arr, denoise_array)
                img = image.array_to_img(denoise_array[0])
                img.save(IMAGES + file_name[:-4] + '_denoised' + file_name[-4:])
                main()
            elif command.strip().lower() in ('n', 'no'):
                main()
            else:
                continue

    # Image denoised. Nothing to do
    else:
        print('Seems like your image denoised.')
        main()


if __name__ == '__main__':
    main()
