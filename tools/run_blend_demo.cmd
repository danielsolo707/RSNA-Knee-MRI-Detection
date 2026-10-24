@echo off
set PY=C:\Users\ASUS\AppData\Local\Programs\Python\Python312\python.exe
set REPO=C:\Users\ASUS\Desktop\DanielEmpire\ML\RSNA Knee Abnormality Detection
"%PY%" "%REPO%\tools\blend_gold.py" --arm "v31=%REPO%\artifacts\v31_runs\gold_oof.csv" --arm "v34=%REPO%\artifacts\v34_remote\raptor_gold_v34f0.npz" --gold "%REPO%\data\raw\train.csv" -o "%REPO%\artifacts\blend_gold_out" 2>&1
echo EXITCODE=%ERRORLEVEL%
