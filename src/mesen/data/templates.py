"""
HTML/CSS UI templates for diverse web applications and components.
Used as foundational baselines for automated UI/UX data synthesis and mutation.
"""

TEMPLATES: dict[str, dict] = {
    "dashboard_analytics": {
        "title": "Analytics Dashboard",
        "category": "dashboard",
        "primary_action_text": "Export Report",
        "html": """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Analytics Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: #f8fafc; color: #1e293b; padding: 24px; }
    header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }
    h1 { font-size: 24px; font-weight: 700; }
    .btn-primary { background: #2563eb; color: #fff; border: none; padding: 10px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .card { background: #fff; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .card-title { font-size: 14px; color: #64748b; margin-bottom: 8px; }
    .card-value { font-size: 28px; font-weight: 700; color: #0f172a; }
    .table-container { background: #fff; border-radius: 12px; border: 1px solid #e2e8f0; padding: 16px; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; text-align: left; }
    th, td { padding: 12px 16px; border-bottom: 1px solid #f1f5f9; }
    th { color: #64748b; font-size: 13px; font-weight: 600; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Platform Overview</h1>
      <p style="color: #64748b; font-size: 14px;">Daily system metrics and user activities</p>
    </div>
    <button class="btn-primary" id="primary-cta">Export Report</button>
  </header>
  <div class="grid">
    <div class="card"><div class="card-title">Total Active Users</div><div class="card-value">24,582</div></div>
    <div class="card"><div class="card-title">System Throughput</div><div class="card-value">1,420 req/s</div></div>
    <div class="card"><div class="card-title">Success Rate</div><div class="card-value">99.98%</div></div>
    <div class="card"><div class="card-title">Avg Latency</div><div class="card-value">4.2 ms</div></div>
  </div>
  <div class="table-container">
    <table>
      <thead>
        <tr><th>Service</th><th>Status</th><th>Latency</th><th>Uptime</th></tr>
      </thead>
      <tbody>
        <tr><td>Auth Gateway</td><td><span style="color:#16a34a; font-weight:600;">Healthy</span></td><td>1.2 ms</td><td>100.0%</td></tr>
        <tr><td>Database Cluster</td><td><span style="color:#16a34a; font-weight:600;">Healthy</span></td><td>2.8 ms</td><td>99.99%</td></tr>
        <tr><td>Queue Broker</td><td><span style="color:#16a34a; font-weight:600;">Healthy</span></td><td>0.9 ms</td><td>100.0%</td></tr>
      </tbody>
    </table>
  </div>
</body>
</html>""",
    },
    "auth_login": {
        "title": "Account Sign In",
        "category": "authentication",
        "primary_action_text": "Sign In to Account",
        "html": """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Account Sign In</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: #f1f5f9; display: flex; align-items: center; justify-content: center; min-height: 100vh; padding: 20px; }
    .auth-card { background: #fff; width: 100%; max-width: 400px; padding: 32px; border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
    h2 { font-size: 22px; margin-bottom: 8px; color: #0f172a; text-align: center; }
    p.subtitle { color: #64748b; font-size: 14px; text-align: center; margin-bottom: 24px; }
    .form-group { margin-bottom: 16px; }
    label { display: block; font-size: 13px; font-weight: 600; color: #334155; margin-bottom: 6px; }
    input[type="text"], input[type="password"] { width: 100%; padding: 10px 14px; border-radius: 8px; border: 1px solid #cbd5e1; font-size: 14px; }
    .btn-submit { width: 100%; background: #0ea5e9; color: #fff; border: none; padding: 12px; border-radius: 8px; font-weight: 600; font-size: 15px; cursor: pointer; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="auth-card">
    <h2>Welcome Back</h2>
    <p class="subtitle">Please enter your credentials to continue</p>
    <div class="form-group">
      <label>Email Address</label>
      <input type="text" value="operator@clinic.internal">
    </div>
    <div class="form-group">
      <label>Password</label>
      <input type="password" value="••••••••••••">
    </div>
    <button class="btn-submit" id="primary-cta">Sign In to Account</button>
  </div>
</body>
</html>""",
    },
    "order_checkout": {
        "title": "Order Checkout",
        "category": "ecommerce",
        "primary_action_text": "Confirm and Pay",
        "html": """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Order Checkout</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: #fafafa; padding: 32px; color: #18181b; }
    .container { max-width: 600px; margin: 0 auto; background: #fff; border: 1px solid #e4e4e7; border-radius: 12px; padding: 24px; }
    h2 { font-size: 20px; font-weight: 700; margin-bottom: 16px; }
    .item-row { display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid #f4f4f5; font-size: 14px; }
    .total-row { display: flex; justify-content: space-between; padding: 16px 0; font-size: 18px; font-weight: 700; color: #09090b; }
    .btn-pay { width: 100%; background: #16a34a; color: #fff; border: none; padding: 14px; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; margin-top: 16px; }
  </style>
</head>
<body>
  <div class="container">
    <h2>Order Summary</h2>
    <div class="item-row"><span>Health Screening Sensor Kit</span><span>$240.00</span></div>
    <div class="item-row"><span>Sterile Disposables Pack (x5)</span><span>$45.00</span></div>
    <div class="item-row"><span>Priority Courier Delivery</span><span>$15.00</span></div>
    <div class="total-row"><span>Total Payable</span><span>$300.00</span></div>
    <button class="btn-pay" id="primary-cta">Confirm and Pay</button>
  </div>
</body>
</html>""",
    },
    "patient_records": {
        "title": "Clinical Patient Records",
        "category": "healthcare",
        "primary_action_text": "Add Measurement Record",
        "html": """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Patient Records</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: #f8fafc; padding: 24px; color: #0f172a; }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
    .btn-record { background: #0284c7; color: #fff; border: none; padding: 10px 16px; border-radius: 8px; font-weight: 600; cursor: pointer; }
    .patient-card { background: #fff; border-radius: 12px; border: 1px solid #e2e8f0; padding: 20px; margin-bottom: 16px; }
    .vitals-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 12px; }
    .vital-item { background: #f0f9ff; padding: 12px; border-radius: 8px; border: 1px solid #bae6fd; }
    .vital-label { font-size: 12px; color: #0369a1; font-weight: 600; }
    .vital-val { font-size: 20px; font-weight: 700; color: #0c4a6e; margin-top: 4px; }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h2 style="font-size: 22px;">Patient Measurement Workspace</h2>
      <p style="color: #64748b; font-size: 13px;">ID: PT-882914 &bull; Status: Active Session</p>
    </div>
    <button class="btn-record" id="primary-cta">Add Measurement Record</button>
  </div>
  <div class="patient-card">
    <h3 style="font-size: 16px; margin-bottom: 4px;">Latest Physiological Observations</h3>
    <div class="vitals-grid">
      <div class="vital-item"><div class="vital-label">Heart Rate</div><div class="vital-val">72 bpm</div></div>
      <div class="vital-item"><div class="vital-label">Systolic BP</div><div class="vital-val">118 mmHg</div></div>
      <div class="vital-item"><div class="vital-label">Diastolic BP</div><div class="vital-val">76 mmHg</div></div>
      <div class="vital-item"><div class="vital-label">SpO2 Level</div><div class="vital-val">98.5%</div></div>
    </div>
  </div>
</body>
</html>""",
    },
    "modal_confirmation": {
        "title": "Action Confirmation Dialog",
        "category": "dialog",
        "primary_action_text": "Confirm Action",
        "html": """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Confirm Action</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: rgba(15, 23, 42, 0.6); display: flex; align-items: center; justify-content: center; min-height: 100vh; padding: 16px; }
    .modal { background: #fff; width: 100%; max-width: 440px; border-radius: 14px; padding: 24px; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1); }
    h3 { font-size: 18px; color: #0f172a; margin-bottom: 8px; }
    p { font-size: 14px; color: #475569; line-height: 1.5; margin-bottom: 24px; }
    .actions { display: flex; justify-content: flex-end; gap: 10px; }
    .btn-secondary { background: #f1f5f9; color: #475569; border: none; padding: 10px 16px; border-radius: 8px; font-weight: 600; cursor: pointer; }
    .btn-primary { background: #dc2626; color: #fff; border: none; padding: 10px 16px; border-radius: 8px; font-weight: 600; cursor: pointer; }
  </style>
</head>
<body>
  <div class="modal">
    <h3>Revoke Operator Credentials</h3>
    <p>Are you sure you want to permanently deactivate this operator key? All connected field terminals will immediately lose synchronization access.</p>
    <div class="actions">
      <button class="btn-secondary">Dismiss</button>
      <button class="btn-primary" id="primary-cta">Confirm Action</button>
    </div>
  </div>
</body>
</html>""",
    },
    "clinical_cancer_screening": {
        "title": "癌症篩檢追蹤個案平台",
        "category": "clinical_screening",
        "primary_action_text": "批次篩檢匯入",
        "html": """<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Hi-Care 癌症篩檢管理平台</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang TC", sans-serif; }
    body { background: #f1f5f9; color: #0f172a; padding: 24px; }
    header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; background: #fff; padding: 16px 24px; border-radius: 12px; border: 1px solid #e2e8f0; }
    h1 { font-size: 20px; font-weight: 700; color: #0f172a; }
    .btn-primary { background: #0284c7; color: #fff; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 600; font-size: 14px; cursor: pointer; }
    .stats-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .stat-card { background: #fff; padding: 18px; border-radius: 10px; border: 1px solid #e2e8f0; }
    .stat-label { font-size: 13px; color: #64748b; margin-bottom: 6px; }
    .stat-num { font-size: 26px; font-weight: 700; color: #0284c7; }
    .filter-bar { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
    .filter-select { padding: 8px 14px; border-radius: 6px; border: 1px solid #cbd5e1; background: #fff; font-size: 13px; }
    .table-card { background: #fff; border-radius: 12px; border: 1px solid #e2e8f0; overflow: hidden; }
    table { width: 100%; border-collapse: collapse; text-align: left; }
    th, td { padding: 14px 18px; border-bottom: 1px solid #f1f5f9; font-size: 14px; }
    th { background: #f8fafc; color: #475569; font-weight: 600; }
    .badge-normal { background: #dcfce7; color: #15803d; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .badge-positive { background: #fee2e2; color: #b91c1c; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .badge-pending { background: #fef9c3; color: #854d0e; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Hi-Care 癌症篩檢個案管理</h1>
      <p style="color: #64748b; font-size: 13px; margin-top: 4px;">雲林海線社區衛生所 &bull; 主責管理員: 許護理長</p>
    </div>
    <button class="btn-primary" id="primary-cta">批次篩檢匯入</button>
  </header>
  <div class="stats-row">
    <div class="stat-card"><div class="stat-label">待篩檢名冊人數</div><div class="stat-num">342</div></div>
    <div class="stat-card"><div class="stat-label">陽性待追蹤個案</div><div class="stat-num" style="color: #dc2626;">18</div></div>
    <div class="stat-card"><div class="stat-label">本月完成率</div><div class="stat-num" style="color: #16a34a;">96.4%</div></div>
    <div class="stat-card"><div class="stat-label">異常個案即時通報</div><div class="stat-num" style="color: #ea580c;">5</div></div>
  </div>
  <div class="filter-bar">
    <select class="filter-select"><option>篩檢項目：全部 (四癌篩檢)</option><option>大腸癌糞便潛血</option><option>口腔黏膜檢查</option><option>乳房攝影</option></select>
    <select class="filter-select"><option>篩檢結果：全部狀態</option><option>正常</option><option>陽性待複查</option><option>檢驗中</option></select>
  </div>
  <div class="table-card">
    <table>
      <thead>
        <tr><th>身分證字號</th><th>受檢者姓名</th><th>性別/年齡</th><th>篩檢類別</th><th>採檢日期</th><th>檢驗狀態</th></tr>
      </thead>
      <tbody>
        <tr><td>P123***789</td><td>陳建國</td><td>男 / 62歲</td><td>大腸癌篩檢</td><td>2026-10-02</td><td><span class="badge-normal">正常 (陰性)</span></td></tr>
        <tr><td>P220***456</td><td>林美惠</td><td>女 / 58歲</td><td>乳房X光攝影</td><td>2026-10-04</td><td><span class="badge-positive">陽性 (需安排複檢)</span></td></tr>
        <tr><td>P121***321</td><td>黃清池</td><td>男 / 71歲</td><td>口腔黏膜檢查</td><td>2026-10-05</td><td><span class="badge-pending">待專科醫師複審</span></td></tr>
      </tbody>
    </table>
  </div>
</body>
</html>""",
    },
    "clinical_patient_profile": {
        "title": "臨床生理數值與長期照護歷程",
        "category": "clinical_ehr",
        "primary_action_text": "新增量測記錄",
        "html": """<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>臨床照護歷程與生理觀測</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang TC", sans-serif; }
    body { background: #f8fafc; color: #0f172a; padding: 24px; }
    .patient-header { background: #fff; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
    .patient-name { font-size: 22px; font-weight: 700; }
    .patient-meta { color: #64748b; font-size: 14px; margin-top: 4px; }
    .btn-record { background: #059669; color: #fff; border: none; padding: 10px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; }
    .vitals-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .vital-tile { background: #fff; padding: 18px; border-radius: 10px; border: 1px solid #e2e8f0; }
    .vital-title { font-size: 13px; color: #64748b; }
    .vital-val { font-size: 24px; font-weight: 700; color: #0f172a; margin-top: 6px; }
    .vital-status { display: inline-block; margin-top: 6px; font-size: 12px; padding: 2px 8px; border-radius: 4px; background: #dcfce7; color: #166534; font-weight: 600; }
    .history-card { background: #fff; border-radius: 12px; border: 1px solid #e2e8f0; padding: 20px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 12px 16px; border-bottom: 1px solid #f1f5f9; font-size: 14px; text-align: left; }
    th { color: #64748b; font-size: 13px; font-weight: 600; }
  </style>
</head>
<body>
  <div class="patient-header">
    <div>
      <div class="patient-name">陳建華 (男，68歲)</div>
      <div class="patient-meta">身分證字號: P120***892 &bull; 健保卡號: 0000-8891-2301 &bull; 慢性病註記: 高血壓第二期</div>
    </div>
    <button class="btn-record" id="primary-cta">新增量測記錄</button>
  </div>
  <div class="vitals-grid">
    <div class="vital-tile"><div class="vital-title">血壓 (收縮/舒張)</div><div class="vital-val">128 / 82 mmHg</div><div class="vital-status">血壓穩定</div></div>
    <div class="vital-tile"><div class="vital-title">心率脈搏</div><div class="vital-val">72 bpm</div><div class="vital-status">竇性心律正常</div></div>
    <div class="vital-tile"><div class="vital-title">空腹血糖</div><div class="vital-val">104 mg/dL</div><div class="vital-status">數值良好</div></div>
    <div class="vital-tile"><div class="vital-title">血液氧氣飽和度</div><div class="vital-val">99.0%</div><div class="vital-status">正常</div></div>
  </div>
  <div class="history-card">
    <h3 style="font-size: 16px; margin-bottom: 12px;">近期巡迴生理量測歷程</h3>
    <table>
      <thead>
        <tr><th>記錄時間</th><th>量測項目</th><th>測量結果</th><th>記錄儀器</th><th>操作人員</th></tr>
      </thead>
      <tbody>
        <tr><td>2026-10-06 09:30</td><td>血壓 / 心率</td><td>128 / 82 mmHg, 72 bpm</td><td>智慧魔鏡社區終端 #02</td><td>社區巡迴護理師</td></tr>
        <tr><td>2026-10-04 14:15</td><td>血糖快篩</td><td>104 mg/dL</td><td>臨床手持生化儀</td><td>林技術員</td></tr>
      </tbody>
    </table>
  </div>
</body>
</html>""",
    },
    "enterprise_data_table": {
        "title": "醫療機構排程與設備維護",
        "category": "enterprise_admin",
        "primary_action_text": "新增維護排程",
        "html": """<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>醫療機構排程與設備維護</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang TC", sans-serif; }
    body { background: #f8fafc; color: #1e293b; padding: 24px; }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
    h2 { font-size: 20px; font-weight: 700; color: #0f172a; }
    .btn-primary { background: #4f46e5; color: #fff; border: none; padding: 10px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; }
    .table-box { background: #fff; border-radius: 12px; border: 1px solid #e2e8f0; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    table { width: 100%; border-collapse: collapse; text-align: left; }
    th, td { padding: 14px 18px; border-bottom: 1px solid #f1f5f9; font-size: 14px; }
    th { background: #f8fafc; font-weight: 600; color: #475569; }
    .pill-active { background: #dcfce7; color: #166534; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; }
    .pill-warn { background: #fef3c7; color: #92400e; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h2>院內生理感測設備調度清單</h2>
      <p style="color: #64748b; font-size: 13px; margin-top: 4px;">列出全院 28 台連網生理監測儀器運作狀態</p>
    </div>
    <button class="btn-primary" id="primary-cta">新增維護排程</button>
  </div>
  <div class="table-box">
    <table>
      <thead>
        <tr><th>設備識別編號</th><th>設備型號名稱</th><th>配置科別</th><th>韌體版本</th><th>上次校正日期</th><th>運作狀態</th></tr>
      </thead>
      <tbody>
        <tr><td>EQ-DEV-2026-001</td><td>Fansee 智慧光學魔鏡 #1</td><td>社區家醫診間</td><td>v2.4.1</td><td>2026-09-15</td><td><span class="pill-active">正常運作</span></td></tr>
        <tr><td>EQ-DEV-2026-002</td><td>Fansee 智慧光學魔鏡 #2</td><td>銀髮長照活動中心</td><td>v2.4.0</td><td>2026-09-18</td><td><span class="pill-active">正常運作</span></td></tr>
        <tr><td>EQ-DEV-2026-003</td><td>攜帶式十二導程心電圖儀</td><td>急診巡迴救護車</td><td>v1.9.8</td><td>2026-08-20</td><td><span class="pill-warn">待定期校正</span></td></tr>
      </tbody>
    </table>
  </div>
</body>
</html>""",
    },
    "multistep_intake_stepper": {
        "title": "受檢者健康問卷與知情同意",
        "category": "patient_intake",
        "primary_action_text": "同意並進入下一步",
        "html": """<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>受檢者知情同意流程</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang TC", sans-serif; }
    body { background: #f8fafc; color: #0f172a; padding: 32px 16px; }
    .form-container { max-width: 640px; margin: 0 auto; background: #fff; border-radius: 14px; border: 1px solid #e2e8f0; padding: 32px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
    .stepper { display: flex; justify-content: space-between; margin-bottom: 28px; border-bottom: 2px solid #f1f5f9; padding-bottom: 16px; }
    .step-item { font-size: 13px; font-weight: 600; color: #94a3b8; }
    .step-item.active { color: #0284c7; }
    h2 { font-size: 20px; font-weight: 700; margin-bottom: 8px; }
    .legal-box { background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; font-size: 13px; color: #475569; line-height: 1.6; max-height: 140px; overflow-y: auto; margin-bottom: 20px; }
    .checkbox-row { display: flex; align-items: center; gap: 10px; margin-bottom: 24px; font-size: 14px; font-weight: 500; }
    .btn-next { width: 100%; background: #0284c7; color: #fff; border: none; padding: 12px; border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer; }
  </style>
</head>
<body>
  <div class="form-container">
    <div class="stepper">
      <div class="step-item">1. 基本身分認證 (已完成)</div>
      <div class="step-item active">2. 知情同意書 (當前步驟)</div>
      <div class="step-item">3. 開始生理量測</div>
    </div>
    <h2>非接觸式光學量測知情同意書</h2>
    <p style="color: #64748b; font-size: 14px; margin-bottom: 16px;">請閱讀以下量測規範並於下方勾選同意</p>
    <div class="legal-box">
      本社區健康魔鏡量測站採用經人體試驗委員會（IRB: CCUREC114111802）核准之電子量測流程。設備將透過環境光感測技術進行非侵入性脈搏波形分析，所得之數據僅供自主健康管理與預防醫學篩檢參考，不可直接作為臨床緊急診斷之單一依據。量測全程採匿名雜湊處理，保障受檢者個人隱私。
    </div>
    <div class="checkbox-row">
      <input type="checkbox" id="agree-check" checked>
      <label for="agree-check">我已詳閱並同意上述量測條款及個人資料去識別化政策</label>
    </div>
    <button class="btn-next" id="primary-cta">同意並進入下一步</button>
  </div>
</body>
</html>""",
    },
    "mobile_health_portal": {
        "title": "個人健康存摺與日常量測",
        "category": "mobile_portal",
        "primary_action_text": "開始今日量測",
        "html": """<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>健康生活存摺</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang TC", sans-serif; }
    body { background: #f8fafc; color: #0f172a; max-width: 440px; margin: 0 auto; min-height: 100vh; padding: 20px 16px 80px 16px; }
    .top-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
    .greeting { font-size: 18px; font-weight: 700; }
    .hero-card { background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); color: #fff; padding: 22px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(2,132,199,0.2); }
    .hero-card h3 { font-size: 18px; margin-bottom: 6px; }
    .hero-card p { font-size: 13px; opacity: 0.9; margin-bottom: 16px; }
    .btn-start { background: #fff; color: #0284c7; border: none; padding: 10px 18px; border-radius: 8px; font-weight: 700; font-size: 14px; cursor: pointer; }
    .grid-menu { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .menu-tile { background: #fff; padding: 16px; border-radius: 12px; border: 1px solid #e2e8f0; text-align: center; }
    .tile-title { font-size: 14px; font-weight: 600; margin-top: 6px; }
  </style>
</head>
<body>
  <div class="top-bar">
    <div>
      <div class="greeting">早安，林淑芬 女士</div>
      <div style="font-size: 12px; color: #64748b;">連續第 14 天進行晨間量測</div>
    </div>
  </div>
  <div class="hero-card">
    <h3>今日生理數據量測</h3>
    <p>早晨血壓與心率尚未登記，請至智慧魔鏡前進行 30 秒感測。</p>
    <button class="btn-start" id="primary-cta">開始今日量測</button>
  </div>
  <div class="grid-menu">
    <div class="menu-tile"><div class="tile-title">歷史血壓趨勢</div></div>
    <div class="menu-tile"><div class="tile-title">衛生所門診掛號</div></div>
    <div class="menu-tile"><div class="tile-title">用藥定時提醒</div></div>
    <div class="menu-tile"><div class="tile-title">癌症篩檢歷程</div></div>
  </div>
</body>
</html>""",
    },
    "telehealth_waiting_room": {
        "title": "遠距視訊診療候診室",
        "category": "telehealth",
        "primary_action_text": "進入視訊診間",
        "html": """<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>遠距雲端視訊門診候診</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang TC", sans-serif; }
    body { background: #f8fafc; color: #0f172a; padding: 24px; display: flex; justify-content: center; }
    .container { max-width: 580px; width: 100%; background: #fff; border-radius: 16px; border: 1px solid #e2e8f0; padding: 28px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
    h2 { font-size: 20px; font-weight: 700; margin-bottom: 8px; }
    .doc-card { display: flex; gap: 16px; background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 12px; padding: 16px; margin: 16px 0; align-items: center; }
    .doc-info h4 { font-size: 16px; color: #0369a1; }
    .doc-info p { font-size: 13px; color: #0284c7; margin-top: 2px; }
    .queue-status { background: #fefce8; border: 1px solid #fef08a; padding: 14px; border-radius: 8px; margin-bottom: 20px; font-size: 14px; color: #854d0e; }
    .check-list { margin-bottom: 24px; font-size: 13px; color: #475569; }
    .check-item { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
    .btn-enter { width: 100%; background: #059669; color: #fff; border: none; padding: 14px; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; }
  </style>
</head>
<body>
  <div class="container">
    <h2>遠距視訊門診候診室</h2>
    <div class="doc-card">
      <div class="doc-info">
        <h4>張明遠 主治醫師</h4>
        <p>台大醫院心臟血管內科 &bull; 預約門診號碼: 09 號</p>
      </div>
    </div>
    <div class="queue-status">
      <strong>目前看診進度: 08 號</strong> (預計等待時間: 3 分鐘)
    </div>
    <div class="check-list">
      <div class="check-item"><span style="color:#16a34a;">✔</span> 視訊鏡頭解析度正常 (1080p)</div>
      <div class="check-item"><span style="color:#16a34a;">✔</span> 麥克風音訊接收良好</div>
      <div class="check-item"><span style="color:#16a34a;">✔</span> 社區魔鏡生理歷史記錄已同步給醫師端</div>
    </div>
    <button class="btn-enter" id="primary-cta">進入視訊診間</button>
  </div>
</body>
</html>""",
    },
    "system_ops_dashboard": {
        "title": "邊緣運算節點與鏡像設備監控",
        "category": "system_ops",
        "primary_action_text": "執行系統健康診斷",
        "html": """<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>邊緣設備維運狀態監控</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang TC", sans-serif; }
    body { background: #0f172a; color: #f8fafc; padding: 24px; }
    .top-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; border-bottom: 1px solid #1e293b; padding-bottom: 16px; }
    h1 { font-size: 22px; font-weight: 700; }
    .btn-diag { background: #3b82f6; color: #fff; border: none; padding: 10px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .card { background: #1e293b; padding: 18px; border-radius: 10px; border: 1px solid #334155; }
    .metric-title { font-size: 13px; color: #94a3b8; }
    .metric-val { font-size: 26px; font-weight: 700; color: #38bdf8; margin-top: 6px; }
    .table-container { background: #1e293b; border-radius: 12px; border: 1px solid #334155; overflow: hidden; }
    table { width: 100%; border-collapse: collapse; text-align: left; }
    th, td { padding: 14px 18px; border-bottom: 1px solid #334155; font-size: 13px; }
    th { background: #0f172a; color: #94a3b8; }
    .status-ok { color: #4ade80; font-weight: 600; }
  </style>
</head>
<body>
  <div class="top-header">
    <div>
      <h1>社區終端邊緣設備集群 (Edge Fleet)</h1>
      <p style="color: #94a3b8; font-size: 13px; margin-top: 4px;">線上即時心跳監控 &bull; 目前就緒節點: 42/42</p>
    </div>
    <button class="btn-diag" id="primary-cta">執行系統健康診斷</button>
  </div>
  <div class="grid">
    <div class="card"><div class="metric-title">邊緣節點連線率</div><div class="metric-val">100.0%</div></div>
    <div class="card"><div class="metric-title">平均推論延遲 (ONNX)</div><div class="metric-val">14.2 ms</div></div>
    <div class="card"><div class="metric-title">光學鏡頭健康指數</div><div class="metric-val" style="color: #4ade80;">99.8%</div></div>
    <div class="card"><div class="metric-title">TLS 憑證剩餘效期</div><div class="metric-val">284 天</div></div>
  </div>
  <div class="table-container">
    <table>
      <thead>
        <tr><th>節點識別碼</th><th>佈署場域</th><th>作業系統版本</th><th>核心溫度</th><th>網路吞吐</th><th>狀態</th></tr>
      </thead>
      <tbody>
        <tr><td>NODE-TW-YL-01</td><td>雲林海線社區關懷據點 A</td><td>Ubuntu 24.04 LTS</td><td>41.2 °C</td><td>12.4 Mbps</td><td><span class="status-ok">● 正常</span></td></tr>
        <tr><td>NODE-TW-YL-02</td><td>東勢鄉老人文康活動中心</td><td>Ubuntu 24.04 LTS</td><td>39.8 °C</td><td>8.9 Mbps</td><td><span class="status-ok">● 正常</span></td></tr>
      </tbody>
    </table>
  </div>
</body>
</html>""",
    },
    "billing_subscription": {
        "title": "診所方案訂閱與授權管理",
        "category": "billing",
        "primary_action_text": "升級專業方案",
        "html": """<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>醫療院所授權與服務方案</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang TC", sans-serif; }
    body { background: #f8fafc; color: #0f172a; padding: 32px 16px; }
    .header { text-align: center; margin-bottom: 32px; }
    h2 { font-size: 24px; font-weight: 700; margin-bottom: 8px; }
    .plans-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; max-width: 900px; margin: 0 auto; }
    .plan-card { background: #fff; border-radius: 14px; border: 1px solid #e2e8f0; padding: 28px; display: flex; flex-direction: column; justify-content: space-between; }
    .plan-card.featured { border: 2px solid #0284c7; box-shadow: 0 8px 16px -2px rgba(2,132,199,0.15); }
    .plan-title { font-size: 18px; font-weight: 700; }
    .plan-price { font-size: 28px; font-weight: 800; color: #0284c7; margin: 16px 0; }
    .feature-list { list-style: none; margin-bottom: 24px; font-size: 14px; color: #475569; }
    .feature-list li { margin-bottom: 10px; }
    .btn-upgrade { background: #0284c7; color: #fff; border: none; padding: 12px; border-radius: 8px; font-weight: 700; cursor: pointer; width: 100%; }
  </style>
</head>
<body>
  <div class="header">
    <h2>智慧醫療數位平台方案</h2>
    <p style="color: #64748b;">為基層診所與社區長照據點提供最完整的遠距追蹤與量測服務</p>
  </div>
  <div class="plans-row">
    <div class="plan-card">
      <div>
        <div class="plan-title">基礎巡迴版</div>
        <div class="plan-price">NT$ 0 <span style="font-size: 14px; font-weight: 500; color: #64748b;">/月</span></div>
        <ul class="feature-list">
          <li>✔ 單台智慧魔鏡連線支援</li>
          <li>✔ 基礎生理測量記錄 (血壓/心率)</li>
          <li>✔ 7 天歷史數據雲端保存</li>
        </ul>
      </div>
      <button style="background:#f1f5f9; color:#475569; border:none; padding:12px; border-radius:8px; font-weight:600;">目前方案</button>
    </div>
    <div class="plan-card featured">
      <div>
        <div style="font-size: 12px; font-weight: 700; color: #0284c7; margin-bottom: 4px;">RECOMMENDED</div>
        <div class="plan-title">專業醫療診所版</div>
        <div class="plan-price">NT$ 1,800 <span style="font-size: 14px; font-weight: 500; color: #64748b;">/月</span></div>
        <ul class="feature-list">
          <li>✔ 無限制邊緣設備與社區站連網</li>
          <li>✔ 癌症篩檢陽性個案自動追蹤通報</li>
          <li>✔ 永久雲端臨床數據與統計匯出</li>
          <li>✔ FHIR 標準格式病歷交換</li>
        </ul>
      </div>
      <button class="btn-upgrade" id="primary-cta">升級專業方案</button>
    </div>
  </div>
</body>
</html>""",
    },
    "knowledge_base_portal": {
        "title": "臨床檢驗作業標準規範 (SOP)",
        "category": "documentation",
        "primary_action_text": "下載最新作業手冊",
        "html": """<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>臨床檢驗技術標準作業程序</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang TC", sans-serif; }
    body { background: #f8fafc; color: #1e293b; display: flex; min-height: 100vh; }
    aside { width: 260px; background: #fff; border-right: 1px solid #e2e8f0; padding: 24px; }
    aside h3 { font-size: 16px; font-weight: 700; margin-bottom: 16px; }
    .nav-link { display: block; padding: 8px 12px; border-radius: 6px; color: #475569; text-decoration: none; font-size: 14px; margin-bottom: 4px; }
    .nav-link.active { background: #f0f9ff; color: #0284c7; font-weight: 600; }
    main { flex: 1; padding: 32px 40px; }
    .doc-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; }
    h1 { font-size: 24px; font-weight: 700; }
    .btn-dl { background: #0284c7; color: #fff; border: none; padding: 10px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; }
    .content-card { background: #fff; border-radius: 12px; border: 1px solid #e2e8f0; padding: 28px; line-height: 1.7; font-size: 15px; color: #334155; }
    .callout { background: #f0fdf4; border-left: 4px solid #16a34a; padding: 16px; border-radius: 6px; margin: 20px 0; font-size: 14px; color: #166534; }
  </style>
</head>
<body>
  <aside>
    <h3>臨床指引手冊</h3>
    <a href="#" class="nav-link active">1.0 生理量測標準操作</a>
    <a href="#" class="nav-link">2.0 癌症篩檢常規複查</a>
    <a href="#" class="nav-link">3.0 異常生理警示處置</a>
    <a href="#" class="nav-link">4.0 去識別化法規合規</a>
  </aside>
  <main>
    <div class="doc-header">
      <div>
        <h1>SOP-2026: 社區非接觸式量測站標準作業規範</h1>
        <p style="color: #64748b; font-size: 13px; margin-top: 4px;">版次: 4.2 &bull; 核准日期: 2026-06-15 &bull; 適用機構: 全國巡迴篩檢團隊</p>
      </div>
      <button class="btn-dl" id="primary-cta">下載最新作業手冊</button>
    </div>
    <div class="content-card">
      <h3 style="font-size: 18px; margin-bottom: 12px; color: #0f172a;">1.1 環境照度與光學量測校準程序</h3>
      <p>現場佈署時，請確保光學魔鏡周圍環境照度維持於 300 至 800 Lux 之間。避免強烈背光直接照射感測鏡面，並請指引受檢者自然注視鏡面十字輔助標線。量測開始前，系統將自動執行 3 秒之光學校正。</p>
      <div class="callout">
        <strong>重要注意事項：</strong>受檢者若剛完成劇烈運動，建議於陰涼處靜坐休息 5 分鐘後再進行血壓量測，以確保量測品質數值達到品質門檻 Q3 (Optimal)。
      </div>
    </div>
  </main>
</body>
</html>""",
    },
    "dark_medical_kiosk": {
        "title": "社區智慧照護量測站",
        "category": "kiosk",
        "primary_action_text": "插入健保卡開始量測",
        "html": """<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>智慧長照生理量測站</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang TC", sans-serif; }
    body { background: #0b0f19; color: #f1f5f9; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; padding: 24px; text-align: center; }
    .kiosk-title { font-size: 32px; font-weight: 800; color: #38bdf8; margin-bottom: 12px; letter-spacing: 1px; }
    .kiosk-subtitle { font-size: 18px; color: #94a3b8; margin-bottom: 40px; }
    .steps-row { display: flex; gap: 24px; justify-content: center; margin-bottom: 48px; flex-wrap: wrap; }
    .step-box { background: #1e293b; border: 2px solid #334155; border-radius: 16px; padding: 28px 24px; width: 220px; }
    .step-icon { font-size: 36px; margin-bottom: 12px; }
    .step-text { font-size: 16px; font-weight: 700; color: #e2e8f0; }
    .btn-touch { background: #059669; color: #fff; border: 2px solid #34d399; padding: 20px 48px; border-radius: 50px; font-size: 22px; font-weight: 800; cursor: pointer; box-shadow: 0 0 30px rgba(16,185,129,0.3); }
  </style>
</head>
<body>
  <div class="kiosk-title">FANSEE 智慧魔鏡長者健康站</div>
  <div class="kiosk-subtitle">三步驟自主量測，為您的健康把關</div>
  <div class="steps-row">
    <div class="step-box"><div class="step-icon">🪪</div><div class="step-text">第一步：插入健保卡</div></div>
    <div class="step-box"><div class="step-icon">🪞</div><div class="step-text">第二步：平視鏡面中心</div></div>
    <div class="step-box"><div class="step-icon">📊</div><div class="step-text">第三步：即時讀取數據</div></div>
  </div>
  <button class="btn-touch" id="primary-cta">插入健保卡開始量測</button>
</body>
</html>""",
    },
}
