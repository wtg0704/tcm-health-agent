@echo off
echo ========================================
echo   中医AI健康顾问 - 系统启动
echo ========================================
echo.

echo [1/2] 启动后端（新窗口，首次约30-60秒加载模型）...
start "后端-API服务" cmd /k "mode con cols=100 lines=20 && cd /d E:\yuyi-cc\tcm_health_agent && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

echo [2/2] 启动前端（新窗口）...
start "前端-用户界面" cmd /k "mode con cols=100 lines=20 && cd /d E:\yuyi-cc\tcm_health_agent && python -m streamlit run frontend/app.py --server.port 8501 --server.headless true"

echo.
echo 启动完成！
echo   后端: http://localhost:8000
echo   前端: http://localhost:8501
echo.
echo 后端窗口显示"启动完成"后即可使用
pause
