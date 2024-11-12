# Code
- code/ISSR_code.ipynb為框架主程式，直接執行即可，檔案中第一個cell可以調整各種參數(參數涵義詳見裡面註解)

# 資料集
位於Dataset中，各檔案說明如下

- 高中英文參考詞彙表: 由大考中心提供，用於衡量candidate難易度，欄位包含如下
    - 詞彙
    - 詞性
    - 難易度
- words_alpha: 包含至今所有的英語詞彙，用於篩選LLM輸出是否是單詞
- processed_gsat_data: 處理過的學測題目(為下方raw_gsat_data的子集)，僅保留必要欄位，用於模型輸入
- raw_gsat_data: 收集大考中心釋出資料後的完整的學測資料集，資料包含如下
    - 題幹
    - correct_option_en: 目標詞彙(正確答案)
    - options_en: 原始四選項
    - options_ch: 收集自三民書局參考書對該選項的翻譯 (Note: 有些年份並未提供中文翻譯)
    - Ph, Pl, P: 不同程度區間學生的答對率 (Ph: 前25%, Pl:後25%, P:鑑別度=Ph-Pl)
    - Pa~Pe: 依序是前1%~20%, 20%~40%, 40%~60%, 60%~80%, 80%~99%程度學生的答對率
    - D1~D4: 不同程度區間學生的鑑別度 (D1計算方式為Pa-Pb, D2為Pb-Pc，以此類推)
    - *_options_rate: 不同程度區間學生的各選項選擇率(t: 整體學生, h: 前25%學生, l:後25%學生)
    - *_option_correct: 不同程度區間學生的正確答案選擇率 (可由上面\*_options_rate推出，這欄位是為了方便分析取用新增的)

# Outputs
存放Experiments Section中各實驗的數據(discusstion跟data analysis的部分則存在Code/discusstion_experiment_code內)，這些都是model raw output，表格分數請搭配Outputs/Evaluator.ipynb重現

# 環境
確保主機已經安裝conda
請執行`conda env create -f environment.yml`即可安裝執行環境


