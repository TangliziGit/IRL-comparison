from PIL import Image
import pickle
import sys

from airplane import Env

gs=Env()
termial=False

while not termial:
    action=input()

    if action=="0":
        img, [x, y], terminal=gs.frame_step(0)
    elif action=='1':
        img, [x, y], terminal=gs.frame_step(1)
    elif action=='s':
        break
    elif action=='r':
        gs.reset(hard=True)
    else:# elif action=='2':
        img, [x, y], terminal=gs.frame_step(2)
    print([x, y], terminal)
    
    if terminal:
        break

# graph={}
# prev=None
# for i in st:
#     # cnt=0
#     # for j in st:
#     #     if i[1]==j[1] and i[0]!=j[0]: cnt+=1
#     # if cnt>1:
#     #     print(i, cnt)
#     if prev==None:
#         prev=i
#         continue
#     if str(prev) not in graph.keys():
#         graph[str(prev)]=i
#     elif graph[str(prev)]!=i and graph[str(prev)][0]!=i[0]:
#         print("Error in", prev, i)
#         print("now", prev, graph[str(prev)])
#         break
#     prev=i

# pickle.dump(op, open("ops.pkl", "wb"))
# pickle.dump(st, open("sts.pkl", "wb"))

