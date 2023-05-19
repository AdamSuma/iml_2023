from PIL import Image
import os

folder_path = '/Users/safefleet/Semester4/iml_2023/task3/dataset/food'

smallest_width = float('inf')
smallest_height = float('inf')
count = 0

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        try:
            with Image.open(file_path) as im:
                width, height = im.size
                smallest_width = min(smallest_width, width)
                smallest_height = min(smallest_height, height)
                count += 1
        except:
            pass

print(f"{smallest_height} x {smallest_width}")
print(count)
