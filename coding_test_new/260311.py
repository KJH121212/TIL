# [홀짝 트리 문제 풀이 요약]
# 1. 구조 파악: 입력은 '포레스트(여러 개의 트리)' 형태이므로, 연결된 노드들끼리 그룹화(BFS)가 필요함.
# 2. 핵심 이론: 
#    - 노드 v가 루트일 때 자식 수 = degree[v]
#    - 노드 v가 루트가 아닐 때 자식 수 = degree[v] - 1
# 3. 홀짝 트리 조건: (번호 % 2) == (자식 수 % 2)를 만족해야 함.
#    - v가 루트로 선정되어 홀짝 트리가 되려면:
#      (1) 루트인 v가 v % 2 == degree[v] % 2 를 만족해야 함.
#      (2) 나머지 모든 노드 u는 u % 2 == (degree[u] - 1) % 2 를 만족해야 함.
# 4. 판별 로직:
#    - v % 2 == degree[v] % 2 인 노드를 'Type-A'라고 하면, 
#      v % 2 == (degree[v] - 1) % 2 인 노드는 'Type-A가 아닌 노드'임. (홀짝이 반대이므로)
#    - 따라서 트리 전체에서 Type-A인 노드가 '딱 1개'라면, 그 녀석을 루트로 삼았을 때만 조건이 완성됨!

from collections import deque # 효율적인 큐 구현을 위해 deque 라이브러리를 가져옵니다.

def solution(nodes, edges):
    # --- 1단계: 그래프 초기화 및 노드별 차수(degree) 계산 ---
    adj = {node: [] for node in nodes} # 각 노드의 인접 노드 정보를 담을 딕셔너리를 초기화합니다.
    degree = {node: 0 for node in nodes} # 각 노드가 가진 간선의 총 개수(차수)를 0으로 초기화합니다.
    
    for u, v in edges: # 주어진 모든 간선 정보를 확인하며 장부를 채웁니다.
        adj[u].append(v) # u번 노드의 친구 목록에 v를 추가합니다.
        adj[v].append(u) # 무방향 그래프이므로 v의 친구 목록에도 u를 추가합니다.
        degree[u] += 1 # u에 연결된 선이 하나 늘어났으므로 차수를 1 증가시킵니다.
        degree[v] += 1 # v에 연결된 선이 하나 늘어났으므로 차수를 1 증가시킵니다.
        
    visited = {node: False for node in nodes} # 각 노드의 방문 여부를 관리하는 딕셔너리입니다.
    total_toe = 0 # 홀짝 트리가 될 수 있는 트리의 총 개수를 저장합니다.
    total_reverse_toe = 0 # 역홀짝 트리가 될 수 있는 트리의 총 개수를 저장합니다.
    
    # --- 2단계: 포레스트 내의 각 트리를 BFS로 하나씩 분리하여 탐색 ---
    for node in nodes: # 모든 노드를 하나씩 살펴보며,
        if not visited[node]: # 아직 어떤 트리에도 속하지 않은(방문 안 한) 노드를 발견하면,
            component = [] # 새 나무 한 그루를 담을 리스트를 만듭니다.
            queue = deque([node]) # 탐색을 위한 큐를 생성하고 시작 노드를 넣습니다.
            visited[node] = True # 시작 노드를 방문 처리합니다.
            
            while queue: # 큐가 빌 때까지(연결된 모든 노드를 찾을 때까지) 반복합니다.
                curr = queue.popleft() # 큐의 맨 앞에서 노드 하나를 꺼냅니다.
                component.append(curr) # 현재 트리의 구성원으로 기록합니다.
                for neighbor in adj[curr]: # 현재 노드와 연결된 이웃들을 확인합니다.
                    if not visited[neighbor]: # 아직 방문하지 않은 이웃이라면,
                        visited[neighbor] = True # 방문 도장을 찍고,
                        queue.append(neighbor) # 다음 차례에 이웃의 이웃을 찾도록 큐에 넣습니다.
            
            # --- 3단계: 분리된 트리 내부에서 홀짝/역홀짝 후보 노드 카운팅 ---
            # oe_type_count: 루트가 되었을 때 홀짝 조건을 만족하는 노드의 수 (Type-A)
            # rev_type_count: 루트가 되었을 때 역홀짝 조건을 만족하는 노드의 수 (Type-B)
            oe_type_count = 0 
            rev_type_count = 0
            
            for v in component: # 현재 트리(나무)에 속한 모든 노드를 하나씩 검사합니다.
                # [홀짝 판별] 노드 번호와 차수의 홀짝성이 같으면 루트 후보입니다.
                if v % 2 == degree[v] % 2:
                    oe_type_count += 1 # 홀짝 트리 루트 후보 카운트 증가
                
                # [역홀짝 판별] 노드 번호와 차수의 홀짝성이 다르면 역루트 후보입니다.
                if v % 2 != degree[v] % 2:
                    rev_type_count += 1 # 역홀짝 트리 루트 후보 카운트 증가
            
            # --- 4단계: 트리 전체의 성립 여부 최종 판별 ---
            # 논리: 트리 내에 '루트일 때 조건을 만족하는 노드'가 딱 하나라면, 
            # 그 노드를 제외한 나머지는 자동으로 '루트가 아닐 때 조건'을 만족하게 됩니다.
            
            # 홀짝 트리가 될 수 있는 트리인지 확인합니다.
            if oe_type_count == 1:
                total_toe += 1 # 조건을 만족하는 루트가 딱 1개 존재하면 정답에 추가합니다.
            
            # 역홀짝 트리가 될 수 있는 트리인지 확인합니다.
            if rev_type_count == 1:
                total_reverse_toe += 1 # 조건을 만족하는 역루트가 딱 1개 존재하면 정답에 추가합니다.
                
    return [total_toe, total_reverse_toe] # 계산된 홀짝 트리와 역홀짝 트리의 개수를 반환합니다.