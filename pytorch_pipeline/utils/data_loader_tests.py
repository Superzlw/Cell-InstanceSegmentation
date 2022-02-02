


def print_out_info_of_dl(dataset, dataloader, mode=None):
    arg_name1 = [v for k, v in locals().items() if k == 'dataset'][0]
    arg_name2 = [a for i, a in locals().items() if i == 'dataloader'][0]
    print(f'----------{arg_name1} for {mode}---------')
    print('the image size:')
    print(dataset[50][0].size())
    print('image info:')
    print(dataset[50][1])
    print('the batch of gt bboxes for an image has the size:')
    print(dataset[50][2].size())
    print('the batch of gt masks for an image has the size:')
    print(dataset[50][3].size())
    print('other two info:')
    print(dataset[50][4], dataset[50][5])
    print(f'----------{arg_name2} for {mode}---------')
    data_iter = iter(dataloader)
    for step in range(1):
        data = next(data_iter)
        print('the image batch size:')
        print(data[0].size())
        print('image info:')
        print(data[1])
        print('the batch of gt bboxes in an image batch has the size:')
        print(data[2].size())
        print('the batch of gt masks in an image batch has the size:')
        print(data[3].size())
        print('other two info:')
        print(data[4], data[5])


