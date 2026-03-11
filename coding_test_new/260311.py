# 홀짝 트리
# https://school.programmers.co.kr/learn/courses/30/lessons/388354

# 풀이
# nodes = [a,b,c, ... ] (이때 내부의 값은 0~400,000이며 중복되지 않음)
# edge = [[a,b],[a,c], ... ] (2차원 배열)
# 포레스트인 경우만 입력으로 제공 된다 = 사이클이 없다 + 노드들이 하나로 연결되어 있지 않을 수도 있다. (트리가 여러개 존재할 수 있다.)

# 문제 풀이 순서
# 1. 포레스트 이기 때문에 여러개 의 트리를 각자의 트리로 나누어 줘야함.
# 2. 이제 트리가 "역홀짝 트리"인지 "홀짝 트리"인지를 확인하는 방법이 필요함

# 1번 문제 풀이
# 인접 리스트 (adj)를 만들어 간선 정보를 저장
# 각 노드의 차수 (degree)를 미리 계산

# 2번 문제풀이
# 전체 node를 순회하며 아직 방문하지 않은 노드를 발견하면 BFS를 시작함
# 해당 BFS를 통해 하나의 tree에 속한 모든 노드를 리스트에 담음

# 3단계
# 각 트리 내에서 다음 두가지 유형의 노드 개수를 셈
# 1. v%2==degree%2  이면 해당 노드는 루트가 되었을 때 홀짝 조건을 만족함
# 2. v%2==(degree-1)%2 이면 해당 노드는 루트가 아닐 때 홀짝 조건을 만족

# 역홀짝의 경우 
# 1. v%2!=degree%2  이면 해당 노드는 루트가 되었을 때 역홀짝 조건을 만족함
# 2. v%2!=(degree-1)%2 이면 해당 노드는 루트가 아닐 때 역홀짝 조건을 만족

# 4단계
# 모든 노드가 루트가 아닐 때 조건을 만족하고, 그중 어떤 노드가 루트일 때 조건도 만족하면 해당 노드를 루트로 삼하 홀짝 트리를 만들 수 있음.
# = 트리 내에서 (루트일때만 만족하는 노드 개수)와 (루트가 아닐 때만 만족하는 노드 개수)를 파악하면 됨.

from collections import deque

def solution(nodes, edges):
    # 1. 그래프 초기화 및 차수 계산
    adj = {node: [] for node in nodes} # 각 노드의 인접 노드 리스트
    degree = {node: 0 for node in nodes} # 각 노드의 간선 연결 개수
    
    for u, v in edges:
        adj[u].append(v) # 무방향 간선 추가
        adj[v].append(u) 
        degree[u] += 1 # 차수 계산
        degree[v] += 1
        
    visited = {node: False for node in nodes} # 방문 여부 체크
    total_toe = 0 # 홀짝 트리 개수
    total_reverse_toe = 0 # 역홀짝 트리 개수
    
    # 2. 포레스트의 각 트리를 BFS로 탐색
    for node in nodes:
        if not visited[node]:
            # 하나의 트리(연결 요소)를 추출합니다.
            component = []
            queue = deque([node])
            visited[node] = True
            
            while queue:
                curr = queue.popleft()
                component.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            # 3. 해당 트리 내에서 조건 만족 노드 개수 파악
            # oe_type: (번호 % 2 == 차수 % 2) 인 노드 수
            # rev_type: (번호 % 2 != 차수 % 2) 인 노드 수
            oe_type_count = 0 
            rev_type_count = 0
            
            for v in component:
                # 노드의 번호와 차수의 홀짝성이 일치하는지 확인
                if v % 2 == degree[v] % 2:
                    oe_type_count += 1
                # 노드의 번호와 차수의 홀짝성이 다른지 확인
                if v % 2 != degree[v] % 2:
                    rev_type_count += 1
            
            # 4. 정답 판별 (핵심 논리)
            # 트리의 노드 수를 N이라 할 때, 
            # 홀짝 트리가 되려면 '루트일 때 조건'을 만족하는 노드가 딱 1개여야 함
            # (나머지 N-1개는 루트가 아닐 때 조건을 만족해야 함)
            if oe_type_count == 1:
                total_toe += 1
            # 역홀짝 트리가 되려면 '루트일 때 조건'을 만족하는 노드가 딱 1개여야 함
            if rev_type_count == 1:
                total_reverse_toe += 1
                
    return [total_toe, total_reverse_toe] # 최종 결과 반환