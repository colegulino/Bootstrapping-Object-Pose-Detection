import modelnet.modelnet as mn

m = mn.modelnet()

train = m.get_train()
test = m.get_test()
valid = m.get_validation()