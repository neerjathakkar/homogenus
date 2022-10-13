

# "idx: 1, prob male:0.5289999842643738, prob female:0.47099998593330383\n"


def parse_line(res_line):
    tokens = res_line.split(",")
    idx = tokens[0].split(":")[1]
    prob_male = round(float(tokens[1].split(":")[1]), 3)
    prob_female = round(float(tokens[2].split(":")[1]), 3)
    # print(prob_male, prob_female)
    return [idx, prob_male, prob_female]

def process_results_file(fname, only_IoU_0 = False, use_conf_thresh=False, confidence_thresh=0.5):
    frame_to_IoU = {}
    frame_data = {}
    f = open(fname, "r")
    curr_frame = None
    for line in f:
        keep_frame_data = False
        tokens = line.split(":")
        if tokens[0] == 'average probabilities':
            pass 
        else:
            if tokens[0] == 'img':
                curr_frame = int(tokens[1].split(".")[0])
            elif tokens[0] == 'iou':
                frame_to_IoU[curr_frame] = float(tokens[1].strip())
            elif tokens[0] == 'idx':
                line_data = parse_line(line)
                if only_IoU_0 and use_conf_thresh:
                    if frame_to_IoU[curr_frame] == 0 and (line_data[1] >= confidence_thresh or line_data[2] >= confidence_thresh):
                        keep_frame_data = True 
                elif only_IoU_0 or use_conf_thresh:
                # print(line_data)
                    if only_IoU_0 and frame_to_IoU[curr_frame] == 0:
                        keep_frame_data = True
                    if use_conf_thresh and (line_data[1] >= confidence_thresh or line_data[2] >= confidence_thresh):
                        keep_frame_data = True
                else:
                    keep_frame_data = True
                if keep_frame_data:
                    if curr_frame in frame_data:
                        frame_data[curr_frame].append(line_data)
                    else:
                        frame_data[curr_frame] = [frame_to_IoU[curr_frame]]
                        frame_data[curr_frame].append(line_data)
                    
    #     print(curr_frame)
    #     print(tokens)
    #     print(line)
    # print(frame_data)

    return frame_data

def get_average_results(frame_data):
    idx_0_total_male = 0
    idx_0_total_female = 0
    idx_0_ct = 0
    idx_1_total_male = 0
    idx_1_total_female = 0
    idx_1_ct = 0
    for frame in frame_data:
        data = frame_data[frame]
        for i, item in enumerate(data):
            if i > 0:
                idx = int(item[0])
                if idx == 0:
                    idx_0_ct += 1
                    idx_0_total_male += item[1]
                    idx_0_total_female += item[2]
                elif idx == 1: 
                    idx_1_ct += 1
                    idx_1_total_male += item[1]
                    idx_1_total_female += item[2]
    idx_0_prob_male = idx_0_total_male/idx_0_ct
    idx_0_prob_female = idx_0_total_female/idx_0_ct
    idx_1_prob_male = idx_1_total_male/idx_1_ct
    idx_1_prob_female = idx_1_total_female/idx_1_ct
    print(f"idx 0 prob male: {idx_0_prob_male}, prob female: {idx_0_prob_female}")
    print(f"idx 1 prob male: {idx_1_prob_male}, prob female: {idx_1_prob_female}")
    


# res_line = "idx: 1, prob male:0.5289999842643738, prob female:0.47099998593330383\n"
# parse_line(res_line)

fnames = ["/home/neerja/swing_dancing/homogenus_out/0HoDSF-2Ldg_img_gendered/results.txt",
          "/home/neerja/swing_dancing/homogenus_out/14ofMLZBGdA_img_gendered/results.txt",
          "/home/neerja/swing_dancing/homogenus_out/1gX5M_BFVp0_img_gendered/results.txt",
          "/home/neerja/swing_dancing/homogenus_out/22heWXVDVbA_img_gendered/results.txt",
          "/home/neerja/swing_dancing/homogenus_out/280WZmWT5Oo_img_gendered/results.txt" ]

for fname in fnames:
    print("\n"+ fname)
    print("original results")
    frame_data = process_results_file(fname, only_IoU_0=False)
    get_average_results(frame_data)
    print("no bbox overlap allowed")
    frame_data = process_results_file(fname, only_IoU_0=True)
    get_average_results(frame_data)
    print("only high confidence results kept")
    frame_data = process_results_file(fname,only_IoU_0=False, use_conf_thresh=True, confidence_thresh=0.95)
    get_average_results(frame_data)
    print("only high confidence results with IoU of 0 kept")
    frame_data = process_results_file(fname,only_IoU_0=True, use_conf_thresh=True, confidence_thresh=0.95)
    get_average_results(frame_data)
