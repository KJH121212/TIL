# https://school.programmers.co.kr/learn/courses/30/lessons/468370 
# 모자이크 중요한 문자 개수 맞추기

def solution(message, spoiler_ranges):
    # 모든 단어의 [텍스트, 시작인덱스, 끝인덱스] 추출
    words_info = []
    temp_word = ""
    start_idx = 0

    for i, char in enumerate(message):
        if char == " ":
            if temp_word:
                words_info.append([temp_word, start_idx, i-1])
                temp_word = ""
        else:
            if temp_word:
                start_idx = i
            temp_word += char
    if temp_word:
        words_info.append([temp_word, start_idx, len(message)-1])

    spoiler_word_indices = []       # 스포일러에 걸친 단어들의 인덱스
    non_spoiler_texts = set()       # 스포일러가 아닌 구간에만 등장한 단어 텍스트

    is_spoiler_word = [False] * len(words_info)

    for i, (text, w_start, w_end) in enumerate(words_info):
        for s_start, s_end in spoiler_ranges:
            # 구간이 겹치는지 확인
            if max(w_start, s_start) <= min(w_end, s_end):
                is_spoiler_word[i] = True
                break
        
        if not is_spoiler_word[i]:
            non_spoiler_texts.add(text)

    important_count = 0
    seen_spoiler_words = set()
    
    for s_start, s_end in spoiler_ranges:
        current_range_words = []
        for i, (text, w_start, w_end) in enumerate(words_info):
            if max(w_start, s_start) <= min(w_end, s_end):
                current_range_words.append(text)
        
        for text in current_range_words:
            if text not in non_spoiler_texts and text not in seen_spoiler_words:
                important_count += 1
            
            seen_spoiler_words.add(text)
            
    return important_count