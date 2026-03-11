# 주요 라이브러리
## 1. collection.deque (큐 구현)
가장 보편적으로 사용되는 양방향 큐  
시작과 끝점에서 데이터를 넣고 빼는 속도가 O(1)로 매우 빠름  
주요 메서드:
- append(x) : 오른쪽에 데이터 삽입
- popleft() : 왼쪽에서 데이터 꺼내기 (가장 먼저 들어온 것)
- appendleft(x) : 왼쪽에 데이터 삽입
- pop() : 오른쪽에서 데이터 꺼내기

```python
from collections import deque

# 1. 데크(deque) 초기화
# 빈 큐를 만들거나, 초기 리스트를 넣어서 만들 수 있습니다.
d = deque([1, 2, 3])

# 2. append(x): 오른쪽(뒤)에 데이터 추가
d.append(4)
print(f"오른쪽에 4 추가: {d}") # deque([1, 2, 3, 4])

# 3. popleft(): 왼쪽(앞)에서 데이터 꺼내기 (큐의 핵심!)
first_item = d.popleft()
print(f"왼쪽에서 꺼낸 값: {first_item}") # 1
print(f"현재 큐 상태: {d}") # deque([2, 3, 4])

# 4. appendleft(x): 왼쪽(앞)에 데이터 추가
d.appendleft(0)
print(f"왼쪽에 0 추가: {d}") # deque([0, 2, 3, 4])

# 5. pop(): 오른쪽(뒤)에서 데이터 꺼내기 (스택처럼 사용 가능)
last_item = d.pop()
print(f"오른쪽에서 꺼낸 값: {last_item}") # 4
print(f"최종 큐 상태: {d}") # deque([0, 2, 3])
```

## 2. headq (우선순위 큐)
데이터 우선순위를 부여하여 가장 낮은 값을 먼저 꺼낼 때 사용합니다.  
완전 이진 트리 형태의 heap 구조를 사용. O(log N)의 속도를 보장  
주오 매서드 : 
- heappush(heap, item) : 데이터를 힙에 집어넣는 함수. 최솟값이 항상 맨 위에 오도록 내부적으로 위치 재배치
- heappop(heap) : 힙에서 가장 작은 값을 꺼내서 반환하는 함수

```python
import heapq

# 1. 빈 리스트 생성 (힙으로 사용 예정)
h = []

# 2. 데이터 넣기 (push)
heapq.heappush(h, 30)
heapq.heappush(h, 10)
heapq.heappush(h, 20)

# 3. 데이터 꺼내기 (pop)
print(heapq.heappop(h)) # 결과: 10 (가장 작은 값)
print(heapq.heappop(h)) # 결과: 20
print(heapq.heappop(h)) # 결과: 30
```

## 3. itertools (조합/순열)
완전 탐색 문제에서 모든 경우의 수를 계산할 때 사용됨.  
반복 가능한 객체에서 요소들을 선택하여 정렬된 결과를 생성  
주요 메서드 : 
- permutations(iterable, r): 순열. 순서를 고려하여 $r$개를 뽑는 모든 경우 (A, B와 B, A는 다름)
- combinations(iterable, r): 조합. 순서 상관없이 $r$개를 뽑는 모든 경우 (A, B와 B, A는 같음)
- product(iterable, repeat=r): 중복 순열. 동일한 요소를 여러 번 뽑는 순열
- combinations_with_replacement(iterable, r): 중복 조합. 동일한 요소를 여러 번 뽑는 조합
```python
from itertools import permutations, combinations

data = ['A', 'B', 'C']

# 1. 순열 (permutations): 3개 중 2개를 뽑아 나열 (순서 O)
perm = list(permutations(data, 2))
print(f"순열 결과: {perm}")
# 결과: [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]

# 2. 조합 (combinations): 3개 중 2개를 선택 (순서 X)
comb = list(combinations(data, 2))
print(f"조합 결과: {comb}")
# 결과: [('A', 'B'), ('A', 'C'), ('B', 'C')]

# 3. 중복 조합 (combinations_with_replacement)
# 본인을 포함해서 2개를 뽑는 모든 경우
comb_res = list(combinations_with_replacement(data, 2))
print(f"중복 조합 결과: {comb_res}")
# 결과: [('A', 'A'), ('A', 'B'), ('A', 'C'), ('B', 'B'), ('B', 'C'), ('C', 'C')]
```


# 핵심 알고리즘
## 그리디
현재 상황에서 지금 당장 좋은 것만 고르는 방법  
나중에 미칠 영향은 고려하지 않으며, '가장 큰 순서대로', '가장 짧은 순서대로' 같은 기준이 중요합니다.

### 핵심 조건
1. 선택 절정: 현재 상태에서 최적의 해를 선택
2. 적절성 검사: 선택한 해가 문제의 제약 조건을 벗어나지 않는지 검사합니다.
3. 해답 검사 : 원래의 문제가 해결되었는지 검사하고, 해결되지 않았다면 다시 선택 단계로 돌아갑니다.

### 대표 예시 (거스름돈 최소화)
```python
# 거슬러 줘야 할 돈
n = 1260
count = 0

# 가장 큰 화폐 단위부터 차례대로 확인 (그리디의 기준: '큰 것부터')
coin_types = [500, 100, 50, 10]

for coin in coin_types:
    # 해당 화폐로 거슬러 줄 수 있는 동전의 개수 세기
    count += n // coin 
    # 거슬러 준 후 남은 돈 업데이트
    n %= coin

print(f"최소 동전 개수: {count}")
```
## 완전 탐색 (Brute Force)
가능한 모든 경우의 수를 일일이 대조하여 정답을 찾는 방식  
이론적으로 모든 문제를 풀 수 있지만, 시간 목잡도가 매우 높다는 단점이 있음.  
### 주요 기법
1. 단순 for 문 : 반복문을 중첩해서 모든 경우를 확인
2. itertools(순열/조합) : 모든 경우의수를 생성할 때 필수
3. 재귀 함수 (recursive) : 자기 자신을 호출하며 모든 경로를 탐색
4. DFS/BFS : 그래프의 모든 노드를 방문할 때 사용 
### 대표 예시 (시각 문제)
"00시 00분 00초부터 N시 59분 59초까지의 모든 시각 중에서 3이 하나라도 포함되는 모든 경우의 수를 구하시오."
```python
h = 5 # 5시까지라고 가정
count = 0

for i in range(h + 1): # 시
    for j in range(60): # 분
        for k in range(60): # 초
            # 매 시각 안에 '3'이 포함되어 있다면 카운트 증가
            if '3' in str(i) + str(j) + str(k):
                count += 1

print(f"3이 포함된 총 경우의 수: {count}")
```
## 깊이 우선 탐색 (DFS)
### 사용하는 경우
1. 모든 경로를 다 탐색해봐야할 때 (백 트래킹)
2. 그래프의 특징을 파악해야 할 때 (사이클 존재 여부, 연결 요소 확인 등)
3. 트리의 깊이가 아주 깊지 않을 때

## 너비 우선 탐색 (BFS)
큐를 이용하여 시작점에서 가까운 노드부터 차례대로 방문

### 사용하는 경우
1. 최단 거리를 찾을 때
2. 층별로 탐색이 필요할 때

## 다이나믹 프로그래밍(DP)
복잡한 문제를 여러개의 작은 하위 문제로 나누어 풀고, 그 결과를 저장해 두었다가 다시 사용하는 최적화 기법  
= **한번 계산한 건 메모장에 적어두고 다시 계산하지 않는다.**

### DP가 필요한 이유 (피보나치 수열 예시)
단순 DFS 방식 사용시 동일한 계산을 수업이 반복하게 됨.  
재귀 방식: F(5)를 구하기 위해 F(3)을 3번, F(2)를 5번 계산함. 숫자가 커질 수록 계산량을 기하급수적으로 늘어남.
DP 방식 : F(2)의 계산을 배열에 저장했다가 추후 필요하면 값을 꺼내서 사용

### DP의 핵심 조건 2가지.
1. 최적 부분 구조  
큰 문제의 답이 작은 문제의 답을 포함하고 있는 구조여야함.  
2. 중복되는 부분 문제  
동일한 작은 문제들이 반복적으로 나타나야함.

### DP를 구하는 2가지 방법
1. Top-Down : 큰 문제를 쪼개며 내려가는 방식 (재귀 + Memorization)
2. Bottom-up: 작은 문제부터 차례대로 쌓아 올리는 방식

### DP 문제 해결 5단계 (팁)
1. DP로 풀수 있는 문제인지 확인
2. 상태(state) 정의 : 배열의 인덱스(dp[i])가 무엇을 의미하는가.
3. 점화식 세우기 : i 번째 상태를 만들기 위햐 i-1이나 i-2 상태를 어떻게 조합할지 식을 만들어야함.
4. 초기값 설정 : dp[0], dp[1] 등 작은 단위의 값ㅇ르 미리 정해둠
5. 구현 : 반복문을 통해 구현
