const express = require('express');
const bodyParser = require('body-parser');
const nodemailer = require('nodemailer');
const cors = require('cors');

const app = express();
const PORT = 3000;

app.use(bodyParser.json());
app.use(cors());

// In-memory store for OTPs
const otpStore = {};

// Configure nodemailer (use your own credentials)
const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: 'altrone23@gmail.com',
    pass: 'pzzn aelo zdcv xwqw'
  }
});

function generateOTP() {
  return Math.floor(100000 + Math.random() * 900000).toString();
}

app.post('/send-otp', (req, res) => {
  const { email } = req.body;
  if (!email) return res.json({ success: false, error: 'Email required.' });
  const otp = generateOTP();
  otpStore[email] = otp;

  // Send OTP email
  const mailOptions = {
    from: 'altrone23@gmail.com',
    to: email,
    subject: 'Your OTP Code',
    text: `Your OTP code is: ${otp}`
  };

  transporter.sendMail(mailOptions, (error, info) => {
    if (error) {
      return res.json({ success: false, error: 'Failed to send OTP.' });
    }
    res.json({ success: true });
  });
});

app.post('/verify-otp', (req, res) => {
  const { email, otp } = req.body;
  if (!email || !otp) return res.json({ success: false, error: 'Email and OTP required.' });
  if (otpStore[email] === otp) {
    delete otpStore[email];
    return res.json({ success: true });
  }
  res.json({ success: false, error: 'Invalid OTP.' });
});

app.listen(PORT, () => {
  console.log(`OTP server running on http://localhost:${PORT}`);
});
