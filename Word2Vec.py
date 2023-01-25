import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as Fun

short_raw_text = '''This is a story that explains how adversity is met differently by different people. There was a girl named Asha who lived with her mother and father in a village. 
One day, her father assigned her a simple task. He took three vessels filled with boiling water. He placed an egg in one vessel, 
a potato in the second vessel, and some tea leaves in the third vessel. He asked Asha to keep an eye on the vessels for about ten to fifteen minutes while the three ingredients in three separate vessels boiled. After the said time, he asked Asha to peel the potato and egg, and strain the tea leaves. Asha was left 
puzzled – she understood her father was trying to explain her something, but she didn’t know what it was.

Her father explained, “All three items were put in the same circumstances. See how they’ve responded differently.” He said that the 
potato turned soft, the egg turned hard, and the tea leaves changed the colour and taste of the water. He further said, “We are all like one of these items. When adversity calls, we respond exactly the way they do. Now, are you a potato, an egg, or tea leaves?”'''.split()
vocab = set(short_raw_text)
vocab_size = len(vocab)

WtI = {w: idx for (idx, w) in enumerate(vocab)}
ItW = {idx: w for (idx, w) in enumerate(vocab)}
print(WtI)
print(ItW)

data = []
labels = []
n = 0
for i in range(2, len(short_raw_text) - 2):
  n +=1
  context_vec = [WtI[short_raw_text[i - 2]], WtI[short_raw_text[i - 1]],
                WtI[short_raw_text[i + 1]], WtI[short_raw_text[i + 2]]]
  context_vec = torch.tensor(context_vec, dtype=torch.long)
  context = Fun.one_hot(context_vec, num_classes = vocab_size)
  context = torch.sum(context, dim = 0, keepdim = True)
  target = torch.tensor(WtI[short_raw_text[i]])
  labels.append(target)
  target = Fun.one_hot(target, num_classes = vocab_size)
  data.append((context, target))
print(n)

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
      super(CBOW, self).__init__()
      self.L1 = nn.Linear(vocab_size, embedding_dim, bias = False)
      self.L3 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
      out = Fun.relu(self.L1(x))
      out = Fun.softmax(self.L3(out), dim = 1)
      return out

model = CBOW(vocab_size, 32)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_func = []
accuracy = []

for i in range(0,800):
  loss = 0
  acc = 0
  for k, (context, target) in enumerate(data):
    output = model(context.float())
    loss += criterion(output.squeeze(0), target.float())
    _, predicted = torch.max(output.data , 1)
    if(predicted == labels[k]):
      acc += 1
  loss_func.append(float(loss))
  accuracy.append(acc/n*100)
  if(i%10 == 0):
    print(loss)
    print(acc/n*100)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

plt.plot(loss_func)
plt.title("CBOW Loss")
plt.show()
plt.plot(accuracy)
plt.title("CBOW Accuracy")
plt.show()

txt = input('Enter a text : ')
txt1 = txt.split()
vec = [WtI[txt1[0]], WtI[txt1[1]], WtI[txt1[2]], WtI[txt1[3]]]
vec = torch.tensor(vec, dtype = torch.long)
cntxt = Fun.one_hot(vec, num_classes = vocab_size)
cntxt = torch.sum(cntxt, dim = 0, keepdim = True)
output = model(cntxt.float())
_, predicted = torch.max(output.data , 1)
print('Predicted : ', ItW[int(predicted)])

for name, param in model.named_parameters():
    if 'L1.weight' in name:
        embed = param.detach().numpy()
        embed = embed.T
        print('Embedding shape : ', embed.shape)
        print(embed)

w1 = input('Enter word-1 : ')
w2 = input('Enter word-2 : ')
w3 = input('Enter word-3 : ')
cos_dist1 = np.dot(embed[WtI[w1]], embed[WtI[w2]])
cos_dist2 = np.dot(embed[WtI[w2]], embed[WtI[w3]])
cos_dist3 = np.dot(embed[WtI[w1]], embed[WtI[w3]])
print('The cosine distance between word-1 and word-2 is : ', cos_dist1)
print('The cosine distance between word-2 and word-3 is : ', cos_dist2)
print('The cosine distance between word-1 and word-3 is : ', cos_dist3)

