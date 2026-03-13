# https://school.programmers.co.kr/learn/courses/30/lessons/468371
# 노란불 신호등

def solution(signals):
    # 각 신호등의 전체 주기(초록+노란+빨간)를 리스트로 저장합니다.
    periods = [sum(s) for s in signals] # 각 신호등의 한 사이클 시간을 구합니다.
    
    limit = 1 # 탐색할 최대 시간 범위를 설정하기 위한 변수입니다.
    for p in periods: # 모든 신호등의 주기를 순회하며,
        limit *= p # 모든 주기의 곱을 구하여 안전한 탐색 한계선을 정합니다. (최대공배수)

    # 1초부터 시작하여 모든 신호등이 노란불인 시점을 찾습니다.
    for t in range(1, limit + 1): # 1초부터 limit초까지 1초씩 증가시키며 확인합니다.
        is_all_yellow = True # 현재 시간 t에 모든 신호등이 노란불인지 체크하는 깃발입니다.
        
        for i in range(len(signals)):
            G, Y, R = signals[i] # i번째 신호등의 초록(G), 노란(Y), 빨간(R) 지속 시간입니다.
            cycle_time = (t - 1) % periods[i] # 현재 시간 t가 해당 신호등 주기에서 몇 번째 초인지 구합니다 (0~period-1).
            
            # 노란불 조건: 초록불 시간(G) 이후부터 초록불+노란불(G+Y) 시간까지입니다.
            # 0-index 기준이므로 G <= cycle_time < G + Y 일 때가 노란불 상태입니다.
            if not (G <= cycle_time < G + Y): # 만약 현재 신호등이 노란불이 아니라면,
                is_all_yellow = False # 깃발을 False로 내리고,
                break # 더 이상 다른 신호등을 검사하지 않고 내부 반복문을 나갑니다.
        
        if is_all_yellow: # 모든 신호등이 노란불 조건을 만족했다면,
            return t # 즉시 해당 시각(t)을 반환하고 종료합니다.
            
    return -1 # limit까지 확인했음에도 찾지 못했다면 불가능한 경우이므로 -1을 반환합니다.