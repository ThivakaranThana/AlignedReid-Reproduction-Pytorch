import pickle
import os.path as osp

with open('/home/niruhan/Dataset/custom_input/partitions.pkl', 'rb') as f:
    partitions = pickle.load(f)

# f = open('/home/niruhan/Dataset/custom_input/partitions.pkl', 'rb')
#
# partitions = pickle.load(f)

print(partitions['test_im_names'])
print(partitions['test_marks'])

partitions['test_im_names'].append("00000004_0001_00000001.jpg")
partitions['test_marks'].append(1)

print(partitions['test_im_names'])
print(partitions['test_marks'])

# partition_file = osp.join(save_dir, 'partitions.pkl')
# save_pickle(partitions, partition_file)

with open('/home/niruhan/Dataset/custom_input/partitions.pkl', 'wb') as f:
    pickle.dump(partitions, f, protocol=2)

person_count = {'00000001' : 3,
                '00000002' : 5}