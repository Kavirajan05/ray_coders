// ...existing code...

document.addEventListener('DOMContentLoaded', function() {
  const loginForm = document.getElementById('loginForm');
  const otpForm = document.getElementById('otpForm');
  const messageDiv = document.getElementById('message');

  loginForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    const email = document.getElementById('loginEmail').value;
    // Proceed to OTP step
    messageDiv.textContent = 'Sending OTP...';
    try {
      const res = await fetch('http://localhost:3000/send-otp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      });
      const data = await res.json();
      if (data.success) {
        messageDiv.textContent = 'OTP sent to your email. Please check your inbox.';
        loginForm.style.display = 'none';
        otpForm.style.display = 'block';
        window.currentLoginEmail = email;
      } else {
        messageDiv.textContent = data.error || 'Error sending OTP.';
      }
    } catch (error) {
      messageDiv.textContent = 'Connection error. Make sure the server is running.';
    }
  });

  otpForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    const email = window.currentLoginEmail;
    const otp = document.getElementById('otp').value;
    messageDiv.textContent = 'Verifying OTP...';
    try {
      const res = await fetch('http://localhost:3000/verify-otp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, otp })
      });
      const data = await res.json();
      if (data.success) {
        messageDiv.textContent = 'OTP verified! Redirecting to dashboard...';
        setTimeout(() => {
          window.location.href = 'index.html';
        }, 1200);
      } else {
        messageDiv.textContent = data.error || 'Invalid OTP.';
      }
    } catch (error) {
      messageDiv.textContent = 'Connection error. Make sure the server is running.';
    }
  });
});
