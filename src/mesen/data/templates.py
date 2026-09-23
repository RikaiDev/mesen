"""
HTML/CSS UI templates for diverse web applications and components.
Used as foundational baselines for automated UI/UX data synthesis and mutation.
"""

from typing import Dict, List

TEMPLATES: Dict[str, Dict] = {
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
</html>"""
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
</html>"""
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
</html>"""
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
</html>"""
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
</html>"""
    }
}
