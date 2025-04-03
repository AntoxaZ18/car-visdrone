from collections import deque

d_len = 80

t = deque(maxlen=d_len)

for i in range(12):
    t.append(i)

batch = 8

step = len(t) / batch

image_batch = []

for i in range(batch):
    image_batch.append(int(i * step))

print(len(image_batch))