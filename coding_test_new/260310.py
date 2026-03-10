# https://school.programmers.co.kr/learn/courses/30/lessons/389480
# 완전 범죄

# 풀이
# 각 물건마다 a가 훔칠지 b가 훔칠지를 결정해야함. 경우의 수가 2^i 으로 i가 40만 되도 경우가 너무 많음
# 따라서 완전 탐색은 불가능함.

# ======= DP의 조건 달성 여부 =======
# 특정 물건까지 훔쳤을 때 a와 b가 남긴 흔적의 합이 같은 경우가 다수 발생한다.
# 따라서 이미계산했던 상황이 반복되므로 "DP" 알고리즘을 이용해야한다.

# ======= 
# 목표 : B의 흔적을 특정 수치만큼 남겼을 때,A의 흔적을 최소화 하는 것
# i번째 물건을 a가 훔치는 경우 B의 흔적은 j 로 동일, A의 흔적 "이전값 + info[i,0]" = "dp[i][j] =dp[i-1][j] + info[i][0]"
# b가 훔치는 경우 B의 흔적이 "dp[i][j+info[i][1]] = min(dp[i][j + info[i][1], dp[i-1][j]])"

def solution(info, n, m): 
    
    INF = float('inf')
    dp = [INF] * m              # dp = [inf,inf,inf, ... , inf]

    dp[0] = 0

    for a_trace, b_trace in info:
        new_dp = [INF] * m
        for j in range(m):
            if dp[j] == INF:
                continue

            # A가 훔치는 경우
            if dp[j] + a_trace < n :
                new_dp[j] = min(new_dp[j], dp[j] + a_trace)
            
            # B가 훔치는 경우
            if j + b_trace < m:
                new_dp[j + b_trace] = min(new_dp[j + b_trace], dp[j])
            
        dp = new_dp
    answer = min(dp)
    return answer

