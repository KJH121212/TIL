# https://school.programmers.co.kr/learn/courses/30/lessons/340198
# 가장 큰 돗자리 깔기

# 풀이 방법
# mats를 내림차순으로 정렬 후, 가장 큰 돗자리부터 넣어서 들어가는지 확인
mats = [5,2,3]
park = [["A", "A", "-1", "B", "B", "B", "B", "-1"],
        ["A", "A", "-1", "B", "B", "B", "B", "-1"], 
        ["-1", "-1", "-1", "-1", "-1", "-1", "-1", "-1"], 
        ["D", "D", "-1", "-1", "-1", "-1", "E", "-1"], 
        ["D", "D", "-1", "-1", "-1", "-1", "-1", "F"], 
        ["D", "D", "-1", "-1", "-1", "-1", "E", "-1"]]

def solution(mats, park):
    mats.sort(reverse=True)
    rows = len(park)        # 공원의 세로 길이
    cols = len(park[0])     # 공원의 가로 길이

    for size in mats:
        for r in range(rows-size+1): # 돗자리가 공원 아래로 벗어나지 않을 범위까지만 반복
            for c in range(cols-size+1): # 돗자리가 공원 옆으로 벗어나지 않을 범위까지만 반복
                is_possible = True
                for i in range(r,r+size): # 돗자리 세로 범위만큼 검사
                    for j in range(c,c+size):
                        if park[i][j] != -1:
                            is_possible = False
                            break
                    if not is_possible: break

                if is_possible:
                    answer = size
    return answer
