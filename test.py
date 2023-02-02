import torch

a = torch.ones(3,3)
mask = ((torch.triu(a)==1).transpose(0,1))

print(a)


print(torch.triu(a))
print(torch.triu(a)==1)
print((torch.triu(a)==1).transpose(0,1))

print('-------------------------')

print(mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)))

b = torch.rand(5,5)

print(b)
# print(b.transpose(0,1))
print(b[:,0::3])
print(b[:,1::3])

t1 = torch.tril(torch.ones(3,3))
t2 = torch.zeros(3,3)


print(t1)
print(t2)
t1[2,0] = 0
print(t2.float().masked_fill(t1==1, 3.0))