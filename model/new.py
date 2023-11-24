N = int(input())

class Place:
    def __init__(self,start,end) -> None:
        self.start = start
        self.end = end

    def __gte__(self,other): # >=
        return self.end>=other.end

    def __lt__(self,other): # <
        return self.end<other.end

    def __repr__(self):  # print
        return f"<Place:[{self.start}-{self.end}]>"

    def against(self,other): # 冲突
        return self.end >= other.start

SLIST = map(int,input().split())
ELIST = map(int,input().split())

P_list = [] 

for s,e in zip(SLIST,ELIST):
    P_list.append(Place(s,e))

P_list.sort()

N_dict = {}

# N_index = [0 for _ in len(N_dict)]

for p_idx,p in enumerate(P_list):
    N_dict[p] = []
    for pn in P_list[p_idx+1:]:
        if not p.against(pn):
            N_dict[p].append(pn)

for k,v in N_dict.items():
    print(f"{k}:{v}")

Route = []

def get_route(node,now):
    # index = P_list.index(node)
    # idx = N_index[index]
    now.append(node)
    if len(now) == 3:
        Route.append(now)
        return
    # if len(N_dict[node]) < idx:
    #     N_index[index] = 0
    #     return
    
    for _n in N_dict[node]:
        print(f"get_route({_n},{now})")
        get_route(_n,now.copy())

for p in P_list:
    get_route(p,[])

print("=======")
for r in Route:
    print(r)
print("=======")
    

    

