# https://school.programmers.co.kr/learn/courses/30/lessons/388352
# 비밀 코드 해독

# 입력 값 : 정수 n, 2차원 정수 배열 q, 응답을 담은 1차원 정수배열 ans
# ex
# n=3, q=[5,n], ans =[n]

# 풀이법

# 예시 결과


from itertools import combinations # 조합을 생성하기 위해 표준 라이브러리를 가져옵니다.

def solution(n, q, ans):
    query_sets = [set(query) for query in q] # 각 시도 정보를 set으로 변환하여 리스트에 담습니다.
    
    answer_count = 0 
    for candidate in combinations(range(1, n + 1), 5): 
        candidate_set = set(candidate) 
        
        is_possible = True
        
        for i in range(len(q)):
            match_count = len(candidate_set & query_sets[i]) # 겹치는 숫자의 개수 계산
            
            if match_count != ans[i]: # 결과가 일치하지 않으면
                is_possible = False # 불가능한 조합으로 표시하고
                break # 더 이상 확인할 필요 없이 다음 후보로 넘어갑니다.
        
        if is_possible:
            answer_count += 1 # 정답 카운트를 1 증가시킵니다.
            
    return answer_count # 모든 시도를 만족하는 가능한 조합의 총 개수를 반환합니다.