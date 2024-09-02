
#检查文件格式
def check_charset(file_path):
    import chardet
    with open(file_path, "rb") as f:
        data = f.read(4)
        charset = chardet.detect(data)['encoding']
    return charset


def get_gene_dict(cancer_type):
    background_path = "../SSN-master/background/"+ cancer_type + "_background.txt"
    # 从文件中读取数据
    with open(background_path, 'r', encoding=check_charset(background_path)) as file:
        content = file.read()

    # 将内容按tab分割为列表
    rows = content.split('\n')

    # 创建一个空数组来存储不重复的项
    unique_items = []

    # 遍历每一行,并添加不重复的项到数组中
    for row in rows:
        items = row.split('\t')
        for item in items:
            if item not in unique_items:
                unique_items.append(item)

    # 将unique_items写入到本地文件
    with open('brca_gene_dict.txt', 'w') as file:
        for item in unique_items:
            file.write(item + '\n')

    # 打印结果
    print(len(unique_items))
    return unique_items

#get_gene_dict("brca")