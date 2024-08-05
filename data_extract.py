import os
import argparse
import sys

def iterate_input(input_path, output_path):
    flag = True
    img = []
    cnt = 1
    slice_count = 1
    with open(input_path, "r") as f:
        for step, i in enumerate(f):
            img.append(i[:-1])
            # import pdb;pdb.set_trace()
            if slice_count == 70:
                
                create_image(img, cnt, output_path)
                slice_count = 0
                img = []
                cnt += 1
            slice_count += 1

    print(cnt)
            
def create_image(img, counter, output_path):
    path = os.path.join(output_path, "image {}.txt".format(counter))
    with open(path, "w+") as f:
        f.write('\n'.join(img))

def get_file_type():
    import pdb;pdb.set_trace()
    parser = argparse.ArgumentParser(description="Get the input type")
    parser.add_argument('--data_type', type=str, default="face")
    args = parser.parse_args()
    file_type = args.data_type
    
    if file_type != "face" and file_type != "digit":
        print("Invalid type")
        sys.exit()
        
    return file_type, args.data_type
        
if __name__ == "__main__":
    file_type, data_type = get_file_type()
    input_path = os.path.join(os.getcwd(),"data", "".join([file_type, "data"]),  "".join([data_type, "images"]))
    output_path = os.path.join(os.getcwd(), "data", "".join(["processed_", file_type]), data_type)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    iterate_input(input_path, output_path)
    