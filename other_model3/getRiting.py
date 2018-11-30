with open("E:\PYworkspace\EETDR\data\ex_stroe_data\\uiRating.txt", "w") as fw:
    with open("E:\PYworkspace\EETDR\data\yelp_restaurant_recursive_entry_sigir\yelp_recursive.entry", "r") as fr:
        for line in fr.readlines():
            eachline = line.strip().split(',')
            ft_sent_pair = eachline[3].strip()
            if ft_sent_pair != '':
                u_idx = eachline[0]
                i_idx = eachline[1]
                over_rating = eachline[2]
                fw.write(eachline[0]+" "+eachline[1]+" "+eachline[2])
                fw.write("\n")
