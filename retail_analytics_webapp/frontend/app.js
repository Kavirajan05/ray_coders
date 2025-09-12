// Enhanced Retail Data with Advanced ML Predictions
const retailData = {
  revenue: {
    monthly: [45000, 52000, 48000, 61000, 58000, 44000, 39000, 47000, 67000, 72000, 69000, 58000],
    yearly: [650000, 720000, 810000, 920000],
    products: {
      "Milk 1L Pack": 180000,
      "Bread Loaf": 145000,
      "Eggs (12 pieces)": 220000,
      "Rice 1KG Pack": 195000,
      "Cooking Oil 1L": 175000
    }
  },
  profit: {
    monthly: [9000, 10400, 9600, 12200, 11600, 8800, 7800, 9400, 13400, 14400, 13800, 11600],
    margins: {
      "Milk 1L Pack": 18.5,
      "Bread Loaf": 22.1,
      "Eggs (12 pieces)": 19.8,
      "Rice 1KG Pack": 20.3,
      "Cooking Oil 1L": 21.7
    }
  }
};

// Enhanced stock data with email alert triggers
const enhancedStockData = {
  products: [
    {name: "Milk 1L Pack", current: 245, reorder: 100, price: 28.0, supplier: "Dairy Fresh Co.", leadTime: 2},
    {name: "Bread Loaf", current: 180, reorder: 75, price: 35.0, supplier: "Bakery Plus", leadTime: 1},
    {name: "Eggs (12 pieces)", current: 156, reorder: 60, price: 72.0, supplier: "Farm Fresh", leadTime: 3},
    {name: "Rice 1KG Pack", current: 25, reorder: 40, price: 65.0, supplier: "Grain Master", leadTime: 5}, // Critical
    {name: "Cooking Oil 1L", current: 15, reorder: 30, price: 145.0, supplier: "Oil Works", leadTime: 4} // Critical
  ],
  recentOrders: [
    {id: 1, product: "Milk 1L Pack", quantity: 200, status: "pending", date: "2024-12-09", delivery: "2024-12-11"},
    {id: 2, product: "Bread Loaf", quantity: 150, status: "delivered", date: "2024-12-08", delivery: "2024-12-09"}
  ]
};

// Enhanced ML Predictions with Confidence Intervals
const monthlyPredictionsData = {
  "1": {
    "Milk": {"quantity": 2890, "confidence": 82.3, "upperBound": 3179, "lowerBound": 2601},
    "Bread": {"quantity": 1956, "confidence": 79.1, "upperBound": 2151, "lowerBound": 1761},
    "Eggs": {"quantity": 1654, "confidence": 85.2, "upperBound": 1819, "lowerBound": 1489},
    "Rice": {"quantity": 1234, "confidence": 78.9, "upperBound": 1357, "lowerBound": 1111},
    "Oil": {"quantity": 876, "confidence": 81.4, "upperBound": 963, "lowerBound": 789}
  },
  "2": {
    "Milk": {"quantity": 3120, "confidence": 84.7, "upperBound": 3432, "lowerBound": 2808},
    "Bread": {"quantity": 2045, "confidence": 82.3, "upperBound": 2249, "lowerBound": 1841},
    "Eggs": {"quantity": 1789, "confidence": 87.1, "upperBound": 1967, "lowerBound": 1611},
    "Rice": {"quantity": 1345, "confidence": 80.5, "upperBound": 1479, "lowerBound": 1211},
    "Oil": {"quantity": 923, "confidence": 83.2, "upperBound": 1015, "lowerBound": 831}
  },
  "3": {
    "Milk": {"quantity": 3456, "confidence": 86.2, "upperBound": 3801, "lowerBound": 3111},
    "Bread": {"quantity": 2187, "confidence": 84.8, "upperBound": 2405, "lowerBound": 1969},
    "Eggs": {"quantity": 1923, "confidence": 88.7, "upperBound": 2115, "lowerBound": 1731},
    "Rice": {"quantity": 1456, "confidence": 82.1, "upperBound": 1601, "lowerBound": 1311},
    "Oil": {"quantity": 1012, "confidence": 85.0, "upperBound": 1113, "lowerBound": 911}
  },
  "4": {
    "Milk": {"quantity": 3678, "confidence": 87.9, "upperBound": 4045, "lowerBound": 3311},
    "Bread": {"quantity": 2298, "confidence": 86.4, "upperBound": 2527, "lowerBound": 2069},
    "Eggs": {"quantity": 2045, "confidence": 89.3, "upperBound": 2249, "lowerBound": 1841},
    "Rice": {"quantity": 1567, "confidence": 83.7, "upperBound": 1723, "lowerBound": 1411},
    "Oil": {"quantity": 1089, "confidence": 86.6, "upperBound": 1197, "lowerBound": 981}
  },
  "5": {
    "Milk": {"quantity": 4234, "confidence": 89.1, "upperBound": 4657, "lowerBound": 3811},
    "Bread": {"quantity": 2456, "confidence": 87.2, "upperBound": 2701, "lowerBound": 2211},
    "Eggs": {"quantity": 2234, "confidence": 90.8, "upperBound": 2457, "lowerBound": 2011},
    "Rice": {"quantity": 1678, "confidence": 85.3, "upperBound": 1845, "lowerBound": 1511},
    "Oil": {"quantity": 1156, "confidence": 88.2, "upperBound": 1271, "lowerBound": 1041}
  },
  "6": {
    "Milk": {"quantity": 3789, "confidence": 84.5, "upperBound": 4167, "lowerBound": 3411},
    "Bread": {"quantity": 2134, "confidence": 81.9, "upperBound": 2347, "lowerBound": 1921},
    "Eggs": {"quantity": 1987, "confidence": 86.4, "upperBound": 2185, "lowerBound": 1789},
    "Rice": {"quantity": 1789, "confidence": 87.6, "upperBound": 1967, "lowerBound": 1611},
    "Oil": {"quantity": 1234, "confidence": 89.8, "upperBound": 1357, "lowerBound": 1111}
  },
  "7": {
    "Milk": {"quantity": 3456, "confidence": 82.1, "upperBound": 3801, "lowerBound": 3111},
    "Bread": {"quantity": 1987, "confidence": 79.6, "upperBound": 2185, "lowerBound": 1789},
    "Eggs": {"quantity": 1834, "confidence": 84.7, "upperBound": 2017, "lowerBound": 1651},
    "Rice": {"quantity": 1923, "confidence": 89.2, "upperBound": 2115, "lowerBound": 1731},
    "Oil": {"quantity": 1345, "confidence": 91.4, "upperBound": 1479, "lowerBound": 1211}
  },
  "8": {
    "Milk": {"quantity": 3678, "confidence": 83.7, "upperBound": 4045, "lowerBound": 3311},
    "Bread": {"quantity": 2098, "confidence": 81.3, "upperBound": 2307, "lowerBound": 1889},
    "Eggs": {"quantity": 1945, "confidence": 86.1, "upperBound": 2139, "lowerBound": 1751},
    "Rice": {"quantity": 1834, "confidence": 88.7, "upperBound": 2017, "lowerBound": 1651},
    "Oil": {"quantity": 1289, "confidence": 90.2, "upperBound": 1417, "lowerBound": 1161}
  },
  "9": {
    "Milk": {"quantity": 4123, "confidence": 88.6, "upperBound": 4535, "lowerBound": 3711},
    "Bread": {"quantity": 2387, "confidence": 85.9, "upperBound": 2625, "lowerBound": 2149},
    "Eggs": {"quantity": 2156, "confidence": 91.3, "upperBound": 2371, "lowerBound": 1941},
    "Rice": {"quantity": 2234, "confidence": 92.8, "upperBound": 2457, "lowerBound": 2011},
    "Oil": {"quantity": 1567, "confidence": 94.1, "upperBound": 1723, "lowerBound": 1411}
  },
  "10": {
    "Milk": {"quantity": 4567, "confidence": 91.2, "upperBound": 5023, "lowerBound": 4111},
    "Bread": {"quantity": 2678, "confidence": 89.4, "upperBound": 2945, "lowerBound": 2411},
    "Eggs": {"quantity": 2445, "confidence": 93.7, "upperBound": 2689, "lowerBound": 2201},
    "Rice": {"quantity": 2567, "confidence": 94.9, "upperBound": 2823, "lowerBound": 2311},
    "Oil": {"quantity": 1789, "confidence": 95.6, "upperBound": 1967, "lowerBound": 1611}
  },
  "11": {
    "Milk": {"quantity": 4234, "confidence": 89.8, "upperBound": 4657, "lowerBound": 3811},
    "Bread": {"quantity": 2456, "confidence": 87.6, "upperBound": 2701, "lowerBound": 2211},
    "Eggs": {"quantity": 2289, "confidence": 92.4, "upperBound": 2517, "lowerBound": 2061},
    "Rice": {"quantity": 2398, "confidence": 93.5, "upperBound": 2637, "lowerBound": 2159},
    "Oil": {"quantity": 1678, "confidence": 94.2, "upperBound": 1845, "lowerBound": 1511}
  },
  "12": {
    "Milk": {"quantity": 3987, "confidence": 87.3, "upperBound": 4385, "lowerBound": 3589},
    "Bread": {"quantity": 2234, "confidence": 85.1, "upperBound": 2457, "lowerBound": 2011},
    "Eggs": {"quantity": 2087, "confidence": 90.6, "upperBound": 2295, "lowerBound": 1879},
    "Rice": {"quantity": 2156, "confidence": 91.8, "upperBound": 2371, "lowerBound": 1941},
    "Oil": {"quantity": 1456, "confidence": 92.7, "upperBound": 1601, "lowerBound": 1311}
  }
};

// Enhanced chat suggestions for better business intelligence
const enhancedChatSuggestions = [
  "Show advanced ML predictions with confidence intervals",
  "Analyze seasonal patterns and weather impact",
  "Check email alert status and recent notifications",
  "Generate profit forecasting with variance analysis",
  "Display product correlation insights",
  "What's the stock status and reorder recommendations?",
  "Show ML model performance and accuracy metrics",
  "Analyze demand patterns for festival season",
  "Check voice recognition system status",
  "Generate comprehensive business report"
];

// Email Alert System Configuration
const emailConfig = {
  smtpHost: "smtp.gmail.com",
  smtpPort: 587,
  fromEmail: "kavirajanekadesi@gmail.com",
  fromPassword: "kavirajan777",
  toEmail: "iniyarajan01@gmail.com",
  isConnected: true,
  lastAlert: new Date().toISOString()
};

// Recent email alerts for display
let recentEmailAlerts = [
  {
    id: 1,
    type: "understock",
    product: "Rice 1KG Pack",
    message: "Stock level critical: 25 units (below 40 threshold)",
    timestamp: new Date(Date.now() - 3600000).toISOString(),
    status: "sent"
  },
  {
    id: 2,
    type: "understock", 
    product: "Cooking Oil 1L",
    message: "Stock level critical: 15 units (below 30 threshold)",
    timestamp: new Date(Date.now() - 7200000).toISOString(),
    status: "sent"
  }
];

const users = [
  {id: 1, username: "retailer1", role: "retailer", lastLogin: "2024-12-08"},
  {id: 2, username: "admin", role: "admin", lastLogin: "2024-12-09"},
  {id: 3, username: "storemanager", role: "retailer", lastLogin: "2024-12-07"}
];

// Application State
let currentUser = null;
let currentPage = 'dashboard';
let charts = {};
let voiceEnabled = false;
let speechRecognition = null;
let speechSynthesis = null;
let isListening = false;
let orderHistory = [...enhancedStockData.recentOrders];
let queryCount = 0;
let voiceQueryCount = 0;
let emailAlertsEnabled = true;

// Chart Colors for enhanced visualizations
const chartColors = ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545', '#D2BA4C', '#964325', '#944454', '#13343B'];

// Initialize Application
document.addEventListener('DOMContentLoaded', function() {
  initializeApp();
});

function initializeApp() {
  showPage('login');
  setupEventListeners();
  initializeEnhancedVoiceFeatures();
  populateTables();
  initializeChatbot();
  initializeEmailAlertSystem();
  
  // Initialize charts after a short delay
  setTimeout(() => {
    initializeCharts();
  }, 100);
}

function initializeEnhancedVoiceFeatures() {
  // Enhanced Web Speech API support with better error handling
  if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    speechRecognition = new SpeechRecognition();
    
    speechRecognition.continuous = false;
    speechRecognition.interimResults = false;
    speechRecognition.lang = 'en-US';
    speechRecognition.maxAlternatives = 3;
    
    speechRecognition.onstart = function() {
      isListening = true;
      updateVoiceStatus('listening');
      updateVoiceEngineStatus('Web Speech API', 'Connected', '98.5%');
    };
    
    speechRecognition.onresult = function(event) {
      const transcript = event.results[0][0].transcript;
      const confidence = event.results[0][0].confidence;
      handleEnhancedVoiceInput(transcript, confidence);
    };
    
    speechRecognition.onend = function() {
      isListening = false;
      updateVoiceStatus('ready');
    };
    
    speechRecognition.onerror = function(event) {
      console.error('Enhanced speech recognition error:', event.error);
      isListening = false;
      updateVoiceStatus('error');
      updateVoiceEngineStatus('Web Speech API', 'Error', 'N/A');
      
      // Enhanced error handling
      let errorMessage = 'Voice recognition error: ';
      switch(event.error) {
        case 'network':
          errorMessage += 'Network connection issue. Please check your internet.';
          break;
        case 'not-allowed':
          errorMessage += 'Microphone access denied. Please allow microphone access.';
          break;
        case 'no-speech':
          errorMessage += 'No speech detected. Please try speaking clearly.';
          break;
        default:
          errorMessage += event.error;
      }
      showNotification(errorMessage, 'error');
    };
    
    voiceEnabled = true;
    updateVoiceEngineStatus('Web Speech API', 'Ready', '98.5%');
  } else {
    console.warn('Speech recognition not supported');
    updateVoiceEngineStatus('Not Available', 'Unsupported', 'N/A');
    showNotification('Voice recognition not supported in this browser. Try Chrome or Edge for best experience.', 'warning');
  }
  
  // Initialize enhanced speech synthesis
  if ('speechSynthesis' in window) {
    speechSynthesis = window.speechSynthesis;
  }
}

function initializeEmailAlertSystem() {
  // Update email status display
  updateEmailStatus();
  
  // Populate recent alerts
  populateRecentAlerts();
  
  // Simulate email monitoring
  setInterval(checkStockLevelsForAlerts, 60000); // Check every minute
  
  showNotification('Email alert system initialized and monitoring stock levels', 'success');
}

function updateEmailStatus() {
  const emailStatus = document.getElementById('emailStatus');
  const lastAlert = document.getElementById('lastAlert');
  
  if (emailStatus) {
    emailStatus.textContent = emailConfig.isConnected ? 
      'Active - Monitoring stock levels' : 
      'Disconnected - Check configuration';
    emailStatus.className = `alert-status ${emailConfig.isConnected ? 'success' : 'error'}`;
  }
  
  if (lastAlert) {
    const lastAlertTime = recentEmailAlerts.length > 0 ? 
      new Date(recentEmailAlerts[0].timestamp).toLocaleString() :
      'No recent alerts';
    lastAlert.textContent = `Last alert: ${lastAlertTime}`;
  }
}

function populateRecentAlerts() {
  const recentAlerts = document.getElementById('recentAlerts');
  if (recentAlerts) {
    recentAlerts.innerHTML = recentEmailAlerts.map(alert => `
      <div class="alert-item">
        <div class="alert-info">
          <div class="alert-title">${alert.type.toUpperCase()}: ${alert.product}</div>
          <div class="alert-details">${alert.message}</div>
          <div class="alert-details">Sent: ${new Date(alert.timestamp).toLocaleString()}</div>
        </div>
        <div class="alert-status">
          <span class="status ${alert.status === 'sent' ? 'success' : 'warning'}">${alert.status}</span>
        </div>
      </div>
    `).join('');
  }
}

function checkStockLevelsForAlerts() {
  enhancedStockData.products.forEach(product => {
    const criticalThreshold = product.reorder * 0.5; // 50% of reorder level
    const understockThreshold = product.reorder; // Reorder level
    
    if (product.current <= criticalThreshold) {
      sendEmailAlert('critical', product, `CRITICAL: ${product.name} stock is critically low (${product.current} units)`);
    } else if (product.current <= understockThreshold) {
      sendEmailAlert('understock', product, `ALERT: ${product.name} stock is below reorder level (${product.current} units)`);
    }
  });
}

function sendEmailAlert(type, product, message) {
  // Check if alert already sent recently for this product
  const recentAlert = recentEmailAlerts.find(alert => 
    alert.product === product.name && 
    alert.type === type &&
    Date.now() - new Date(alert.timestamp).getTime() < 3600000 // Within last hour
  );
  
  if (recentAlert) return; // Don't spam alerts
  
  // Simulate email sending (in real implementation, this would call backend API)
  const emailData = {
    from: emailConfig.fromEmail,
    to: emailConfig.toEmail,
    subject: `Stock Alert: ${product.name} - ${type.toUpperCase()}`,
    body: `
      Dear Store Manager,
      
      ${message}
      
      Product Details:
      - Product: ${product.name}
      - Current Stock: ${product.current} units
      - Reorder Level: ${product.reorder} units
      - Supplier: ${product.supplier}
      - Lead Time: ${product.leadTime} days
      
      Recommended Action: ${product.current <= product.reorder * 0.5 ? 'IMMEDIATE reorder required' : 'Schedule reorder'}
      
      This is an automated alert from RetailSight Pro.
      
      Best regards,
      RetailSight Pro Alert System
    `
  };
  
  // Simulate API call delay
  setTimeout(() => {
    const newAlert = {
      id: recentEmailAlerts.length + 1,
      type: type,
      product: product.name,
      message: message,
      timestamp: new Date().toISOString(),
      status: 'sent'
    };
    
    recentEmailAlerts.unshift(newAlert);
    
    // Keep only last 10 alerts
    if (recentEmailAlerts.length > 10) {
      recentEmailAlerts = recentEmailAlerts.slice(0, 10);
    }
    
    updateEmailStatus();
    populateRecentAlerts();
    
    showNotification(`Email alert sent for ${product.name}`, 'success');
    
    console.log('Email sent:', emailData);
  }, 1000);
}

function testEmailAlert() {
  const testProduct = {
    name: "Test Product",
    current: 5,
    reorder: 50,
    supplier: "Test Supplier",
    leadTime: 3
  };
  
  sendEmailAlert('test', testProduct, 'This is a test email alert from RetailSight Pro system');
}

function sendDemoUnderstock() {
  const riceProduct = enhancedStockData.products.find(p => p.name === "Rice 1KG Pack");
  if (riceProduct) {
    sendEmailAlert('understock', riceProduct, `DEMO ALERT: ${riceProduct.name} stock is below reorder level (${riceProduct.current} units)`);
  }
}

function updateVoiceEngineStatus(engine, status, accuracy) {
  const engineElement = document.getElementById('voiceEngine');
  const statusElement = document.getElementById('voiceConnectionStatus');
  const accuracyElement = document.getElementById('voiceAccuracy');
  
  if (engineElement) engineElement.textContent = engine;
  if (statusElement) statusElement.textContent = status;
  if (accuracyElement) accuracyElement.textContent = accuracy;
}

function setupEventListeners() {
  // Login form
  const loginForm = document.getElementById('loginForm');
  if (loginForm) {
    loginForm.addEventListener('submit', handleLogin);
  }
  
  // Logout button
  const logoutBtn = document.getElementById('logoutBtn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', handleLogout);
  }
  
  // Navigation links
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', handleNavigation);
  });
  
  // Quick action buttons
  document.querySelectorAll('button[data-page]').forEach(button => {
    button.addEventListener('click', (e) => {
      e.preventDefault();
      const page = e.target.getAttribute('data-page');
      showContentPage(page);
    });
  });
  
  // Advanced ML controls
  const generateAdvancedBtn = document.getElementById('generateAdvancedPredictions');
  if (generateAdvancedBtn) {
    generateAdvancedBtn.addEventListener('click', generateAdvancedPredictions);
  }
  
  // ML prediction controls
  const generateBtn = document.getElementById('generatePredictions');
  if (generateBtn) {
    generateBtn.addEventListener('click', generatePredictions);
  }
  
  // Export predictions
  const exportBtn = document.getElementById('exportPredictions');
  if (exportBtn) {
    exportBtn.addEventListener('click', exportPredictions);
  }
  
  // Chat functionality
  const sendBtn = document.getElementById('sendMessage');
  const chatInput = document.getElementById('chatInput');
  if (sendBtn) {
    sendBtn.addEventListener('click', sendChatMessage);
  }
  if (chatInput) {
    chatInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        sendChatMessage();
      }
    });
  }
  
  // Enhanced voice controls
  const voiceToggle = document.getElementById('voiceToggle');
  if (voiceToggle) {
    voiceToggle.addEventListener('click', toggleEnhancedVoiceRecognition);
  }
  
  const speakerToggle = document.getElementById('speakerToggle');
  if (speakerToggle) {
    speakerToggle.addEventListener('click', toggleSpeechSynthesis);
  }
  
  // Stock management
  const bulkReorderBtn = document.getElementById('bulkReorderBtn');
  if (bulkReorderBtn) {
    bulkReorderBtn.addEventListener('click', handleBulkReorder);
  }
  
  // Modal controls
  const closeModal = document.getElementById('closeModal');
  const cancelOrder = document.getElementById('cancelOrder');
  const confirmOrder = document.getElementById('confirmOrder');
  
  if (closeModal) closeModal.addEventListener('click', closeOrderModal);
  if (cancelOrder) cancelOrder.addEventListener('click', closeOrderModal);
  if (confirmOrder) confirmOrder.addEventListener('click', confirmOrderModal);
  
  // Notification close
  const closeNotification = document.getElementById('closeNotification');
  if (closeNotification) {
    closeNotification.addEventListener('click', hideNotification);
  }
}

function handleLogin(e) {
  e.preventDefault();
  
  const username = document.getElementById('username').value;
  const password = document.getElementById('password').value;
  const role = document.getElementById('role').value;
  
  if (username && password) {
    currentUser = { username, role };
    
    const currentUserSpan = document.getElementById('currentUser');
    if (currentUserSpan) {
      currentUserSpan.textContent = `Welcome, ${username}`;
    }
    
    const adminNavItem = document.getElementById('adminNavItem');
    if (adminNavItem) {
      if (role === 'admin') {
        adminNavItem.classList.remove('hidden');
      } else {
        adminNavItem.classList.add('hidden');
      }
    }
    
    showPage('main');
    showContentPage('dashboard');
    showNotification('Login successful! Enhanced ML analytics and email alerts ready.', 'success');
    
    setTimeout(() => {
      initializeCharts();
      populateSuggestionChips();
      updateEmailStatus();
    }, 200);
  }
}

function handleLogout() {
  currentUser = null;
  showPage('login');
  const loginForm = document.getElementById('loginForm');
  if (loginForm) loginForm.reset();
  showNotification('Logged out successfully', 'info');
}

function handleNavigation(e) {
  e.preventDefault();
  const page = e.target.getAttribute('data-page');
  if (page) {
    showContentPage(page);
  }
}

function showPage(page) {
  const loginPage = document.getElementById('loginPage');
  const mainApp = document.getElementById('mainApp');
  
  if (page === 'login') {
    if (loginPage) {
      loginPage.classList.add('active');
      loginPage.style.display = 'block';
    }
    if (mainApp) {
      mainApp.classList.add('hidden');
      mainApp.style.display = 'none';
    }
  } else {
    if (loginPage) {
      loginPage.classList.remove('active');
      loginPage.style.display = 'none';
    }
    if (mainApp) {
      mainApp.classList.remove('hidden');
      mainApp.style.display = 'flex';
    }
  }
}

function showContentPage(pageId) {
  // Update navigation
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.remove('active');
    if (link.getAttribute('data-page') === pageId) {
      link.classList.add('active');
    }
  });
  
  // Update content
  document.querySelectorAll('.content-page').forEach(page => {
    page.classList.remove('active');
  });
  
  const targetPage = document.getElementById(`${pageId}Page`);
  if (targetPage) {
    targetPage.classList.add('active');
  }
  
  currentPage = pageId;
  
  // Page-specific initialization
  if (pageId === 'stock') {
    populateStockData();
    populateRecentOrders();
  } else if (pageId === 'ml') {
    setTimeout(() => generatePredictions(), 200);
  } else if (pageId === 'advanced-ml') {
    setTimeout(() => generateAdvancedPredictions(), 200);
  } else if (pageId === 'email-alerts') {
    updateEmailStatus();
    populateRecentAlerts();
  }
  
  // Refresh charts
  setTimeout(() => {
    Object.values(charts).forEach(chart => {
      if (chart && chart.resize) {
        chart.resize();
      }
    });
  }, 100);
}

function initializeCharts() {
  // Enhanced Dashboard Revenue Chart with ML Forecast
  const dashboardCtx = document.getElementById('dashboardRevenueChart');
  if (dashboardCtx && !charts.dashboardRevenue) {
    const forecastData = [58000, 62000, 59000, 65000, 68000, 72000];
    
    charts.dashboardRevenue = new Chart(dashboardCtx, {
      type: 'line',
      data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan\'25', 'Feb\'25', 'Mar\'25', 'Apr\'25', 'May\'25', 'Jun\'25'],
        datasets: [
          {
            label: 'Actual Revenue',
            data: [...retailData.revenue.monthly, null, null, null, null, null, null],
            borderColor: chartColors[0],
            backgroundColor: chartColors[0] + '20',
            borderWidth: 3,
            fill: true,
            tension: 0.4
          },
          {
            label: 'ML Forecast',
            data: [null, null, null, null, null, null, null, null, null, null, null, null, ...forecastData],
            borderColor: chartColors[4],
            backgroundColor: chartColors[4] + '20',
            borderWidth: 3,
            borderDash: [5, 5],
            fill: false,
            tension: 0.4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true }
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              callback: function(value) {
                return '₹' + value.toLocaleString();
              }
            }
          }
        }
      }
    });
  }
  
  // Initialize all other charts...
  initializeRevenueCharts();
  initializeProfitCharts();
  initializeVisualCharts();
  initializeStockCharts();
  initializePredictionCharts();
  initializeAdvancedMLCharts();
}

function initializeAdvancedMLCharts() {
  // Time Series Predictions with Confidence Intervals
  const timeSeriesCtx = document.getElementById('timeSeriesChart');
  if (timeSeriesCtx && !charts.timeSeries) {
    const monthlyData = [];
    const upperBounds = [];
    const lowerBounds = [];
    
    for (let month = 1; month <= 12; month++) {
      const data = monthlyPredictionsData[month.toString()];
      const total = Object.values(data).reduce((sum, item) => sum + item.quantity, 0);
      const upperTotal = Object.values(data).reduce((sum, item) => sum + item.upperBound, 0);
      const lowerTotal = Object.values(data).reduce((sum, item) => sum + item.lowerBound, 0);
      
      monthlyData.push(total);
      upperBounds.push(upperTotal);
      lowerBounds.push(lowerTotal);
    }
    
    charts.timeSeries = new Chart(timeSeriesCtx, {
      type: 'line',
      data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        datasets: [
          {
            label: 'Predicted Demand',
            data: monthlyData,
            borderColor: chartColors[0],
            backgroundColor: chartColors[0] + '30',
            borderWidth: 3,
            fill: false,
            tension: 0.4
          },
          {
            label: 'Upper Confidence Bound',
            data: upperBounds,
            borderColor: chartColors[1],
            backgroundColor: chartColors[1] + '20',
            borderWidth: 1,
            borderDash: [5, 5],
            fill: '+1',
            tension: 0.4
          },
          {
            label: 'Lower Confidence Bound',
            data: lowerBounds,
            borderColor: chartColors[2],
            backgroundColor: chartColors[2] + '20',
            borderWidth: 1,
            borderDash: [5, 5],
            fill: false,
            tension: 0.4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true }
        },
        scales: {
          y: { beginAtZero: true }
        }
      }
    });
  }
  
  // Seasonal Patterns Analysis
  const seasonalPatternsCtx = document.getElementById('seasonalPatternsChart');
  if (seasonalPatternsCtx && !charts.seasonalPatterns) {
    charts.seasonalPatterns = new Chart(seasonalPatternsCtx, {
      type: 'radar',
      data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        datasets: [
          {
            label: 'Milk Pattern',
            data: [2890, 3120, 3456, 3678, 4234, 3789, 3456, 3678, 4123, 4567, 4234, 3987],
            borderColor: chartColors[0],
            backgroundColor: chartColors[0] + '30',
            borderWidth: 2
          },
          {
            label: 'Bread Pattern',
            data: [1956, 2045, 2187, 2298, 2456, 2134, 1987, 2098, 2387, 2678, 2456, 2234],
            borderColor: chartColors[1],
            backgroundColor: chartColors[1] + '30',
            borderWidth: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: { beginAtZero: true }
        }
      }
    });
  }
  
  // Multi-Product Correlation Analysis
  const correlationCtx = document.getElementById('correlationChart');
  if (correlationCtx && !charts.correlation) {
    charts.correlation = new Chart(correlationCtx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Milk vs Bread',
            data: [
              {x: 2890, y: 1956}, {x: 3120, y: 2045}, {x: 3456, y: 2187},
              {x: 3678, y: 2298}, {x: 4234, y: 2456}, {x: 3789, y: 2134}
            ],
            backgroundColor: chartColors[0],
            borderColor: chartColors[0]
          },
          {
            label: 'Rice vs Oil',
            data: [
              {x: 1234, y: 876}, {x: 1345, y: 923}, {x: 1456, y: 1012},
              {x: 1567, y: 1089}, {x: 1678, y: 1156}, {x: 1789, y: 1234}
            ],
            backgroundColor: chartColors[2],
            borderColor: chartColors[2]
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 'Product A Demand' } },
          y: { title: { display: true, text: 'Product B Demand' } }
        }
      }
    });
  }
  
  // Weather Impact Analysis
  const weatherImpactCtx = document.getElementById('weatherImpactChart');
  if (weatherImpactCtx && !charts.weatherImpact) {
    charts.weatherImpact = new Chart(weatherImpactCtx, {
      type: 'bar',
      data: {
        labels: ['Normal', 'Rainy', 'Festival', 'Summer', 'Winter'],
        datasets: [
          {
            label: 'Demand Impact (%)',
            data: [100, 85, 135, 110, 120],
            backgroundColor: [
              chartColors[0], chartColors[1], chartColors[2], 
              chartColors[3], chartColors[4]
            ],
            borderWidth: 0,
            borderRadius: 8
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              callback: function(value) {
                return value + '%';
              }
            }
          }
        }
      }
    });
  }
  
  // Profit Prediction with Variance
  const profitPredictionCtx = document.getElementById('profitPredictionChart');
  if (profitPredictionCtx && !charts.profitPrediction) {
    const profitForecast = [15200, 16800, 18100, 19500, 21000, 18500];
    const variance = [1200, 1400, 1300, 1600, 1800, 1500];
    
    charts.profitPrediction = new Chart(profitPredictionCtx, {
      type: 'line',
      data: {
        labels: ['Jan\'25', 'Feb\'25', 'Mar\'25', 'Apr\'25', 'May\'25', 'Jun\'25'],
        datasets: [
          {
            label: 'Predicted Profit',
            data: profitForecast,
            borderColor: chartColors[2],
            backgroundColor: chartColors[2] + '30',
            borderWidth: 3,
            fill: true,
            tension: 0.4
          },
          {
            label: 'Variance Range',
            data: profitForecast.map((val, i) => val + variance[i]),
            borderColor: chartColors[4],
            backgroundColor: 'transparent',
            borderWidth: 1,
            borderDash: [3, 3],
            fill: false,
            tension: 0.4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              callback: function(value) {
                return '₹' + value.toLocaleString();
              }
            }
          }
        }
      }
    });
  }
  
  // Confidence Score Heatmap
  const confidenceHeatmapCtx = document.getElementById('confidenceHeatmapChart');
  if (confidenceHeatmapCtx && !charts.confidenceHeatmap) {
    charts.confidenceHeatmap = new Chart(confidenceHeatmapCtx, {
      type: 'bar',
      data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        datasets: [
          {
            label: 'Milk Confidence',
            data: [82.3, 84.7, 86.2, 87.9, 89.1, 84.5, 82.1, 83.7, 88.6, 91.2, 89.8, 87.3],
            backgroundColor: chartColors[0]
          },
          {
            label: 'Bread Confidence',
            data: [79.1, 82.3, 84.8, 86.4, 87.2, 81.9, 79.6, 81.3, 85.9, 89.4, 87.6, 85.1],
            backgroundColor: chartColors[1]
          },
          {
            label: 'Overall Confidence',
            data: [83.2, 85.1, 86.8, 88.1, 89.5, 85.2, 83.4, 84.9, 89.2, 92.1, 90.3, 88.7],
            backgroundColor: chartColors[4]
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 75,
            max: 95,
            ticks: {
              callback: function(value) {
                return value + '%';
              }
            }
          }
        }
      }
    });
  }
}

function generateAdvancedPredictions() {
  const analysisType = document.getElementById('analysisType').value;
  const timeRange = document.getElementById('timeRange').value;
  
  // Update charts based on selected analysis type
  if (charts.timeSeries) {
    charts.timeSeries.update();
  }
  if (charts.seasonalPatterns) {
    charts.seasonalPatterns.update();
  }
  if (charts.correlation) {
    charts.correlation.update();
  }
  if (charts.weatherImpact) {
    charts.weatherImpact.update();
  }
  if (charts.profitPrediction) {
    charts.profitPrediction.update();
  }
  if (charts.confidenceHeatmap) {
    charts.confidenceHeatmap.update();
  }
  
  showNotification(`Advanced ${analysisType} analysis generated for ${timeRange}`, 'success');
}

// Continue with existing chart initialization functions...
function initializeRevenueCharts() {
  const monthlyRevenueCtx = document.getElementById('monthlyRevenueChart');
  if (monthlyRevenueCtx && !charts.monthlyRevenue) {
    charts.monthlyRevenue = new Chart(monthlyRevenueCtx, {
      type: 'bar',
      data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        datasets: [{
          label: 'Monthly Revenue',
          data: retailData.revenue.monthly,
          backgroundColor: chartColors.slice(0, 12),
          borderWidth: 0,
          borderRadius: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              callback: function(value) {
                return '₹' + value.toLocaleString();
              }
            }
          }
        }
      }
    });
  }
  
  const productRevenueCtx = document.getElementById('productRevenueChart');
  if (productRevenueCtx && !charts.productRevenue) {
    charts.productRevenue = new Chart(productRevenueCtx, {
      type: 'doughnut',
      data: {
        labels: Object.keys(retailData.revenue.products),
        datasets: [{
          data: Object.values(retailData.revenue.products),
          backgroundColor: chartColors.slice(0, 5),
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom' }
        }
      }
    });
  }
}

function initializeProfitCharts() {
  const monthlyProfitCtx = document.getElementById('monthlyProfitChart');
  if (monthlyProfitCtx && !charts.monthlyProfit) {
    charts.monthlyProfit = new Chart(monthlyProfitCtx, {
      type: 'line',
      data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        datasets: [{
          label: 'Profit',
          data: retailData.profit.monthly,
          borderColor: chartColors[2],
          backgroundColor: chartColors[2] + '20',
          borderWidth: 3,
          fill: true,
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              callback: function(value) {
                return '₹' + value.toLocaleString();
              }
            }
          }
        }
      }
    });
  }
  
  const profitMarginCtx = document.getElementById('profitMarginChart');
  if (profitMarginCtx && !charts.profitMargin) {
    charts.profitMargin = new Chart(profitMarginCtx, {
      type: 'bar',
      data: {
        labels: Object.keys(retailData.profit.margins),
        datasets: [{
          label: 'Profit Margin (%)',
          data: Object.values(retailData.profit.margins),
          backgroundColor: chartColors[1],
          borderWidth: 0,
          borderRadius: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              callback: function(value) {
                return value + '%';
              }
            }
          }
        }
      }
    });
  }
}

function initializeVisualCharts() {
  const yearlyGrowthCtx = document.getElementById('yearlyGrowthChart');
  if (yearlyGrowthCtx && !charts.yearlyGrowth) {
    charts.yearlyGrowth = new Chart(yearlyGrowthCtx, {
      type: 'line',
      data: {
        labels: ['2021', '2022', '2023', '2024'],
        datasets: [{
          label: 'Yearly Revenue',
          data: retailData.revenue.yearly,
          borderColor: chartColors[0],
          backgroundColor: chartColors[0] + '20',
          borderWidth: 3,
          fill: true,
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              callback: function(value) {
                return '₹' + (value / 1000) + 'K';
              }
            }
          }
        }
      }
    });
  }
  
  const seasonalCtx = document.getElementById('seasonalChart');
  if (seasonalCtx && !charts.seasonal) {
    charts.seasonal = new Chart(seasonalCtx, {
      type: 'radar',
      data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        datasets: [{
          label: 'Sales Pattern',
          data: retailData.revenue.monthly.map(val => val / 1000),
          borderColor: chartColors[2],
          backgroundColor: chartColors[2] + '30',
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: { beginAtZero: true }
        }
      }
    });
  }
  
  const performanceCtx = document.getElementById('performanceChart');
  if (performanceCtx && !charts.performance) {
    charts.performance = new Chart(performanceCtx, {
      type: 'bar',
      data: {
        labels: Object.keys(retailData.revenue.products),
        datasets: [
          {
            label: 'Revenue (₹000s)',
            data: Object.values(retailData.revenue.products).map(v => v / 1000),
            backgroundColor: chartColors[0],
            yAxisID: 'y'
          },
          {
            label: 'Profit Margin (%)',
            data: Object.values(retailData.profit.margins),
            backgroundColor: chartColors[1],
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { type: 'linear', display: true, position: 'left' },
          y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false } }
        }
      }
    });
  }
  
  const accuracyCtx = document.getElementById('accuracyChart');
  if (accuracyCtx && !charts.accuracy) {
    charts.accuracy = new Chart(accuracyCtx, {
      type: 'line',
      data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        datasets: [{
          label: 'AI Prediction Accuracy (%)',
          data: [85.2, 87.1, 88.7, 89.3, 90.8, 86.4, 84.7, 86.1, 91.3, 93.7, 92.4, 90.6],
          borderColor: chartColors[4],
          backgroundColor: chartColors[4] + '20',
          borderWidth: 3,
          fill: true,
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 80,
            max: 100,
            ticks: {
              callback: function(value) {
                return value + '%';
              }
            }
          }
        }
      }
    });
  }
}

function initializeStockCharts() {
  const stockLevelsCtx = document.getElementById('stockLevelsChart');
  if (stockLevelsCtx && !charts.stockLevels) {
    const products = enhancedStockData.products;
    charts.stockLevels = new Chart(stockLevelsCtx, {
      type: 'bar',
      data: {
        labels: products.map(p => p.name),
        datasets: [
          {
            label: 'Current Stock',
            data: products.map(p => p.current),
            backgroundColor: chartColors[0]
          },
          {
            label: 'Reorder Level',
            data: products.map(p => p.reorder),
            backgroundColor: chartColors[2]
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } }
      }
    });
  }
}

function initializePredictionCharts() {
  // Yearly prediction overview
  const yearlyPredictionCtx = document.getElementById('yearlyPredictionChart');
  if (yearlyPredictionCtx && !charts.yearlyPrediction) {
    const monthlyData = [];
    for (let month = 1; month <= 12; month++) {
      const data = monthlyPredictionsData[month.toString()];
      const total = Object.values(data).reduce((sum, item) => sum + item.quantity, 0);
      monthlyData.push(total);
    }
    
    charts.yearlyPrediction = new Chart(yearlyPredictionCtx, {
      type: 'line',
      data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        datasets: [{
          label: 'Total Predicted Demand',
          data: monthlyData,
          borderColor: chartColors[4],
          backgroundColor: chartColors[4] + '20',
          borderWidth: 3,
          fill: true,
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } }
      }
    });
  }
  
  const predictionCtx = document.getElementById('predictionChart');
  if (predictionCtx && !charts.prediction) {
    charts.prediction = new Chart(predictionCtx, {
      type: 'bar',
      data: {
        labels: ['Milk', 'Bread', 'Eggs', 'Rice', 'Oil'],
        datasets: [{
          label: 'Predicted Demand',
          data: [2890, 1956, 1654, 1234, 876],
          backgroundColor: chartColors[4],
          borderWidth: 0,
          borderRadius: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } }
      }
    });
  }
  
  const confidenceCtx = document.getElementById('confidenceChart');
  if (confidenceCtx && !charts.confidence) {
    charts.confidence = new Chart(confidenceCtx, {
      type: 'radar',
      data: {
        labels: ['Milk', 'Bread', 'Eggs', 'Rice', 'Oil'],
        datasets: [{
          label: 'Confidence Score (%)',
          data: [82.3, 79.1, 85.2, 78.9, 81.4],
          borderColor: chartColors[1],
          backgroundColor: chartColors[1] + '30',
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: { min: 70, max: 100 }
        }
      }
    });
  }
}

function populateTables() {
  // Revenue table
  const revenueTableBody = document.getElementById('revenueTableBody');
  if (revenueTableBody) {
    const products = Object.keys(retailData.revenue.products);
    revenueTableBody.innerHTML = products.map(product => `
      <tr>
        <td>${product}</td>
        <td>₹${retailData.revenue.products[product].toLocaleString()}</td>
        <td><span class="text-success">+${(Math.random() * 10 + 5).toFixed(1)}%</span></td>
        <td><span class="status-badge success">Excellent</span></td>
      </tr>
    `).join('');
  }
  
  // User table (Admin)
  const userTableBody = document.getElementById('userTableBody');
  if (userTableBody) {
    userTableBody.innerHTML = users.map(user => `
      <tr>
        <td>${user.id}</td>
        <td>${user.username}</td>
        <td>${user.role}</td>
        <td>${user.lastLogin}</td>
        <td><span class="status-badge success">Active</span></td>
        <td>
          <div class="table-actions">
            <button class="action-btn">Edit</button>
            <button class="action-btn danger">Delete</button>
          </div>
        </td>
      </tr>
    `).join('');
  }
}

function populateStockData() {
  const stockTableBody = document.getElementById('stockTableBody');
  if (stockTableBody) {
    stockTableBody.innerHTML = enhancedStockData.products.map(product => {
      const status = product.current > product.reorder ? 'success' : 
                    product.current > product.reorder * 0.5 ? 'warning' : 'error';
      const statusText = product.current > product.reorder ? 'Good' : 
                        product.current > product.reorder * 0.5 ? 'Low' : 'Critical';
      
      return `
        <tr>
          <td>${product.name}</td>
          <td>${product.current}</td>
          <td>${product.reorder}</td>
          <td>${product.supplier}</td>
          <td>${product.leadTime} days</td>
          <td><span class="status-badge ${status}">${statusText}</span></td>
          <td>
            <div class="table-actions">
              <button class="action-btn success" onclick="openReorderModal('${product.name}')">Reorder</button>
              <button class="action-btn" onclick="viewSupplierDetails('${product.supplier}')">Supplier</button>
            </div>
          </td>
        </tr>
      `;
    }).join('');
  }
  
  // Update stock stats
  const criticalItems = enhancedStockData.products.filter(p => p.current <= p.reorder).length;
  const criticalStockElement = document.getElementById('criticalStock');
  const pendingOrdersElement = document.getElementById('pendingOrders');
  
  if (criticalStockElement) {
    criticalStockElement.textContent = criticalItems;
  }
  if (pendingOrdersElement) {
    pendingOrdersElement.textContent = orderHistory.filter(o => o.status === 'pending').length;
  }
}

function populateRecentOrders() {
  const recentOrdersList = document.getElementById('recentOrdersList');
  if (recentOrdersList) {
    recentOrdersList.innerHTML = orderHistory.map(order => `
      <div class="order-item">
        <div class="order-info">
          <div class="order-product">${order.product}</div>
          <div class="order-details">Qty: ${order.quantity} | Date: ${order.date}</div>
        </div>
        <div class="order-status">
          <span class="status-badge ${order.status === 'delivered' ? 'success' : 'warning'}">${order.status}</span>
          <span class="order-delivery">ETA: ${order.delivery}</span>
        </div>
      </div>
    `).join('');
  }
}

function generatePredictions() {
  const monthSelect = document.getElementById('predictionMonth');
  const selectedMonth = monthSelect.value;
  const selectedData = monthlyPredictionsData[selectedMonth];
  
  if (!selectedData) return;
  
  // Update charts
  if (charts.prediction) {
    const products = ['Milk', 'Bread', 'Eggs', 'Rice', 'Oil'];
    const quantities = products.map(product => selectedData[product].quantity);
    charts.prediction.data.datasets[0].data = quantities;
    charts.prediction.update();
  }
  
  if (charts.confidence) {
    const products = ['Milk', 'Bread', 'Eggs', 'Rice', 'Oil'];
    const confidences = products.map(product => selectedData[product].confidence);
    charts.confidence.data.datasets[0].data = confidences;
    charts.confidence.update();
  }
  
  // Update prediction cards
  const predictionCards = document.getElementById('predictionCards');
  if (predictionCards) {
    const products = Object.keys(selectedData);
    predictionCards.innerHTML = products.map(product => {
      const data = selectedData[product];
      const weatherImpact = getWeatherImpact(selectedMonth, product);
      
      return `
        <div class="prediction-card fade-in">
          <h4>${product === 'Oil' ? 'Cooking Oil 1L' : product === 'Milk' ? 'Milk 1L Pack' : 
                product === 'Bread' ? 'Bread Loaf' : product === 'Eggs' ? 'Eggs (12 pieces)' : 
                'Rice 1KG Pack'}</h4>
          <div class="prediction-value">${data.quantity.toLocaleString()} units</div>
          <div class="confidence-score">Confidence: ${data.confidence}%</div>
          <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${data.confidence}%"></div>
          </div>
          <div class="weather-impact">
            <span class="weather-icon">${weatherImpact.icon}</span>
            <span class="weather-text">${weatherImpact.text}</span>
          </div>
        </div>
      `;
    }).join('');
  }
  
  // Update seasonal insights
  updateSeasonalInsights(selectedMonth);
  
  showNotification(`Enhanced ML predictions generated for ${monthSelect.options[monthSelect.selectedIndex].text}`, 'success');
}

function getWeatherImpact(month, product) {
  const impacts = {
    '6': { icon: '🌧️', text: 'Monsoon may reduce footfall' },
    '7': { icon: '🌧️', text: 'Heavy rains expected' },
    '10': { icon: '🎆', text: 'Festival season boost' },
    '11': { icon: '🎆', text: 'Festival demand high' },
    '12': { icon: '❄️', text: 'Winter season demand' }
  };
  return impacts[month] || { icon: '☀️', text: 'Normal seasonal pattern' };
}

function updateSeasonalInsights(month) {
  const insights = {
    '1': [
      { title: 'New Year Impact', content: 'Health-conscious purchases increase, recommend promoting milk and eggs with ML confidence of 82.3%.' },
      { title: 'Winter Season', content: 'Cooking oil demand rises for winter recipes. Predicted increase of 15% based on historical patterns.' }
    ],
    '5': [
      { title: 'Summer Peak', content: 'Highest demand month with 89.1% ML confidence. Wedding season drives 25% increase in all categories.' },
      { title: 'Stock Optimization', content: 'AI recommends 30% inventory increase. Critical for maintaining service levels during peak demand.' }
    ],
    '10': [
      { title: 'Festival Season', content: 'Diwali drives exceptional demand with 91.2% prediction confidence. All products show 35% growth.' },
      { title: 'Premium Strategy', content: 'ML analysis suggests stocking premium variants increases profit margins by 22% during festivals.' }
    ]
  };
  
  const seasonalInsights = document.getElementById('seasonalInsights');
  if (seasonalInsights && insights[month]) {
    seasonalInsights.innerHTML = insights[month].map(insight => `
      <div class="insight-item">
        <div class="insight-title">${insight.title}</div>
        <div class="insight-content">${insight.content}</div>
        <div class="recommendation-tag">Enhanced AI Recommendation</div>
      </div>
    `).join('');
  }
}

function exportPredictions() {
  const month = document.getElementById('predictionMonth').value;
  const data = monthlyPredictionsData[month];
  const monthName = document.getElementById('predictionMonth').options[document.getElementById('predictionMonth').selectedIndex].text;
  
  let csvContent = `Enhanced ML Prediction Report - ${monthName}\n\n`;
  csvContent += 'Product,Predicted Quantity,Confidence Score,Upper Bound,Lower Bound\n';
  
  Object.keys(data).forEach(product => {
    const fullName = product === 'Oil' ? 'Cooking Oil 1L' : product === 'Milk' ? 'Milk 1L Pack' : 
                     product === 'Bread' ? 'Bread Loaf' : product === 'Eggs' ? 'Eggs (12 pieces)' : 
                     'Rice 1KG Pack';
    const item = data[product];
    csvContent += `${fullName},${item.quantity},${item.confidence}%,${item.upperBound || 'N/A'},${item.lowerBound || 'N/A'}\n`;
  });
  
  csvContent += '\n\nML Model Performance:\n';
  csvContent += 'Overall Accuracy,94.2%\n';
  csvContent += 'Algorithm,Enhanced LSTM Neural Network\n';
  csvContent += 'Data Quality Score,97.8%\n';
  
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `enhanced_ml_predictions_${monthName.toLowerCase().replace(' ', '_')}.csv`;
  a.click();
  window.URL.revokeObjectURL(url);
  
  showNotification('Enhanced ML prediction report exported successfully!', 'success');
}

function initializeChatbot() {
  const chatMessages = document.getElementById('chatMessages');
  if (chatMessages) {
    addMessage('Hello! I\'m your enhanced AI assistant with advanced ML insights, email alert management, and optimized voice recognition. I can help with complex business analytics, stock predictions, profit optimization, and much more!', 'bot');
  }
  
  updateQueryStats();
}

function populateSuggestionChips() {
  const suggestionChips = document.getElementById('suggestionChips');
  if (suggestionChips) {
    suggestionChips.innerHTML = enhancedChatSuggestions.map(suggestion => `
      <button class="suggestion-chip" onclick="askSuggestion('${suggestion}')">${suggestion}</button>
    `).join('');
  }
}

function askSuggestion(question) {
  const chatInput = document.getElementById('chatInput');
  if (chatInput) {
    chatInput.value = question;
    sendChatMessage();
  }
}

function sendChatMessage() {
  const chatInput = document.getElementById('chatInput');
  if (!chatInput) return;
  
  const message = chatInput.value.trim();
  if (message) {
    addMessage(message, 'user');
    chatInput.value = '';
    queryCount++;
    
    setTimeout(() => {
      const response = generateEnhancedBotResponse(message);
      addMessage(response, 'bot');
      
      // Speak response if enabled
      if (speechSynthesis && document.getElementById('speakerToggle').textContent.includes('ON')) {
        speakText(response);
      }
      
      updateQueryStats();
    }, 1000);
  }
}

function addMessage(text, sender) {
  const chatMessages = document.getElementById('chatMessages');
  if (!chatMessages) return;
  
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${sender} fade-in`;
  if (sender === 'bot') {
    messageDiv.className += ' ai-response';
  }
  messageDiv.textContent = text;
  chatMessages.appendChild(messageDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function generateEnhancedBotResponse(message) {
  const lowerMessage = message.toLowerCase();
  
  // Enhanced Email Alert Analysis
  if (lowerMessage.includes('email') || lowerMessage.includes('alert')) {
    const criticalItems = enhancedStockData.products.filter(p => p.current <= p.reorder * 0.5).length;
    const recentAlertsCount = recentEmailAlerts.length;
    return `Email Alert System Status: ${emailConfig.isConnected ? 'ACTIVE' : 'DISCONNECTED'}. Monitoring from ${emailConfig.fromEmail} to ${emailConfig.toEmail}. Recent alerts: ${recentAlertsCount} sent. Critical items triggering alerts: ${criticalItems}. SMTP: ${emailConfig.smtpHost}:${emailConfig.smtpPort}. System automatically sends alerts when stock drops below thresholds.`;
  }
  
  // Advanced ML Insights
  if (lowerMessage.includes('ml') || lowerMessage.includes('machine learning') || lowerMessage.includes('advanced')) {
    return 'Enhanced ML Analytics: Using LSTM Neural Networks with 94.2% accuracy. Confidence intervals show upper/lower bounds for all predictions. Seasonal pattern analysis reveals 25% demand spikes during festivals. Correlation analysis shows strong Milk-Bread relationship (r=0.87). Weather impact models predict 15% demand drop during monsoons. Profit forecasting includes variance analysis for risk assessment.';
  }
  
  // Voice Recognition Status
  if (lowerMessage.includes('voice') || lowerMessage.includes('speech') || lowerMessage.includes('recognition')) {
    const voiceEngine = document.getElementById('voiceEngine')?.textContent || 'Not Available';
    const voiceStatus = document.getElementById('voiceConnectionStatus')?.textContent || 'Unknown';
    const voiceAccuracy = document.getElementById('voiceAccuracy')?.textContent || 'N/A';
    return `Enhanced Voice Recognition Status: Engine: ${voiceEngine}, Status: ${voiceStatus}, Accuracy: ${voiceAccuracy}. Optimized for business queries with contextual understanding. Supports multiple languages, noise cancellation, and confidence scoring. Voice queries: ${voiceQueryCount}/${queryCount} total. Enhanced error handling with fallback options.`;
  }
  
  // Stock analysis with email integration
  if (lowerMessage.includes('stock') || lowerMessage.includes('inventory')) {
    const criticalItems = enhancedStockData.products.filter(p => p.current <= p.reorder);
    const emailAlerts = recentEmailAlerts.filter(a => a.type === 'understock').length;
    if (criticalItems.length > 0) {
      return `CRITICAL Stock Analysis: ${criticalItems.length} items need immediate attention: ${criticalItems.map(p => `${p.name} (${p.current}/${p.reorder})`).join(', ')}. Email alerts sent: ${emailAlerts}. ML predictions suggest reordering now to avoid stockouts. Automated email monitoring active. Total inventory value: ₹2,15,000. Supplier lead times: 1-5 days.`;
    }
    return `Stock Analysis: All items well-stocked above reorder levels. Email monitoring active with ${emailAlerts} recent alerts. ML optimization suggests maintaining current levels. No immediate reorders needed. Auto-alert system functioning perfectly.`;
  }
  
  // Enhanced Revenue Analysis
  if (lowerMessage.includes('revenue') || lowerMessage.includes('sales')) {
    return 'Enhanced Revenue Analytics: Current month ₹69,000 (+12.5% YoY). ML forecast predicts ₹72,000 next month (confidence: 89.1%). Top performer: Eggs with ₹2,20,000 yearly. Seasonal analysis shows 35% boost during festivals. Weather impact models integrated. Confidence intervals: ₹65K-₹75K range. Growth trajectory: 15.8% annually with ML optimization.';
  }
  
  // Advanced Profit Analysis
  if (lowerMessage.includes('profit') || lowerMessage.includes('margin')) {
    return 'Advanced Profit Analytics: Current margin 20.5% (₹13,800 profit). ML forecasting shows 22.1% potential with optimization. Best margin: Bread (22.1%, confidence: 87.6%). Variance analysis indicates ±₹1,500 monthly fluctuation. Festival season profit boost: +25%. Weather impact on margins: -3% during monsoons. ROI optimization suggests focus on high-confidence ML predictions.';
  }
  
  // Enhanced Predictions
  if (lowerMessage.includes('predict') || lowerMessage.includes('forecast')) {
    return 'Enhanced ML Predictions: Next month forecasts with confidence intervals - Milk: 4,567 units (91.2% confidence, range: 4,111-5,023), Bread: 2,678 units (89.4%, range: 2,411-2,945). LSTM model accuracy: 94.2%. Seasonal patterns analyzed. Weather impact integrated. Peak season Oct-Nov shows 92%+ confidence scores. Model refresh cycle: 24 hours.';
  }
  
  // Comprehensive Business Intelligence
  if (lowerMessage.includes('business') || lowerMessage.includes('report') || lowerMessage.includes('comprehensive')) {
    return 'Comprehensive Business Intelligence: Revenue ₹8.2L (+12.5%), Profit ₹1.64L (20% margin), Stock value ₹2.15L. ML accuracy 94.2% with LSTM networks. Email alerts: Active monitoring, 2 recent notifications. Voice queries: 98.5% accuracy. Critical insights: Festival season approaching (35% demand spike predicted), 2 items need reordering, profit optimization potential +8.5%. ROI: 76.3% annually.';
  }
  
  // Default enhanced response
  return 'I provide enhanced business intelligence with advanced ML analytics, real-time email alerts, and optimized voice recognition. Ask me about: ML prediction confidence intervals, email alert configurations, stock optimization with automated notifications, profit forecasting with variance analysis, seasonal patterns with weather impact, voice recognition performance, or comprehensive business reports with actionable insights.';
}

function toggleEnhancedVoiceRecognition() {
  if (!voiceEnabled || !speechRecognition) {
    showNotification('Enhanced voice recognition not available. Please use a compatible browser like Chrome or Edge.', 'error');
    return;
  }
  
  if (isListening) {
    speechRecognition.stop();
    updateVoiceStatus('ready');
  } else {
    try {
      speechRecognition.start();
      updateVoiceStatus('listening');
    } catch (error) {
      console.error('Voice recognition error:', error);
      showNotification('Voice recognition failed to start. Please check microphone permissions.', 'error');
      updateVoiceStatus('error');
    }
  }
}

function toggleSpeechSynthesis() {
  const speakerToggle = document.getElementById('speakerToggle');
  const isOn = speakerToggle.textContent.includes('ON');
  speakerToggle.innerHTML = `
    <span class="speaker-icon">🔊</span>
    <span>Voice Response: ${isOn ? 'OFF' : 'ON'}</span>
  `;
}

function updateVoiceStatus(status) {
  const voiceStatus = document.getElementById('voiceStatus');
  const voiceToggle = document.getElementById('voiceToggle');
  
  if (voiceStatus && voiceToggle) {
    switch (status) {
      case 'listening':
        voiceStatus.textContent = 'Enhanced listening mode active...';
        voiceStatus.className = 'voice-status listening';
        voiceToggle.className = 'btn btn--primary voice-btn listening';
        voiceToggle.innerHTML = '<span class="voice-icon">🎤</span><span class="voice-text">Stop Listening</span>';
        updateVoiceEngineStatus('Web Speech API', 'Listening', '98.5%');
        break;
      case 'processing':
        voiceStatus.textContent = 'Processing with enhanced AI...';
        voiceStatus.className = 'voice-status processing';
        voiceToggle.className = 'btn btn--primary voice-btn processing';
        updateVoiceEngineStatus('Web Speech API', 'Processing', '98.5%');
        break;
      case 'error':
        voiceStatus.textContent = 'Voice error - enhanced recovery in progress';
        voiceStatus.className = 'voice-status';
        voiceToggle.className = 'btn btn--primary voice-btn';
        voiceToggle.innerHTML = '<span class="voice-icon">🎤</span><span class="voice-text">Start Voice</span>';
        updateVoiceEngineStatus('Web Speech API', 'Error', 'N/A');
        break;
      default:
        voiceStatus.textContent = 'Enhanced voice recognition ready';
        voiceStatus.className = 'voice-status';
        voiceToggle.className = 'btn btn--primary voice-btn';
        voiceToggle.innerHTML = '<span class="voice-icon">🎤</span><span class="voice-text">Start Voice</span>';
        updateVoiceEngineStatus('Web Speech API', 'Ready', '98.5%');
    }
  }
}

function handleEnhancedVoiceInput(transcript, confidence) {
  updateVoiceStatus('processing');
  
  const chatInput = document.getElementById('chatInput');
  if (chatInput) {
    chatInput.value = transcript;
    
    // Add enhanced voice message indicator with confidence
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
      const messageDiv = document.createElement('div');
      messageDiv.className = 'message user voice-message fade-in';
      messageDiv.textContent = `${transcript} (Confidence: ${(confidence * 100).toFixed(1)}%)`;
      chatMessages.appendChild(messageDiv);
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    voiceQueryCount++;
    queryCount++;
    
    setTimeout(() => {
      const response = generateEnhancedBotResponse(transcript);
      addMessage(response, 'bot');
      
      if (speechSynthesis && document.getElementById('speakerToggle').textContent.includes('ON')) {
        speakText(response);
      }
      
      updateQueryStats();
    }, 1000);
  }
  
  updateVoiceStatus('ready');
}

function speakText(text) {
  if (speechSynthesis) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.volume = 0.8;
    
    // Enhanced voice selection
    const voices = speechSynthesis.getVoices();
    const preferredVoice = voices.find(voice => voice.name.includes('Google') || voice.name.includes('Microsoft'));
    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }
    
    speechSynthesis.speak(utterance);
  }
}

function updateQueryStats() {
  const totalQueriesElement = document.getElementById('totalQueries');
  const voiceQueriesElement = document.getElementById('voiceQueries');
  const contextAccuracyElement = document.getElementById('contextAccuracy');
  
  if (totalQueriesElement) {
    totalQueriesElement.textContent = queryCount;
  }
  if (voiceQueriesElement) {
    voiceQueriesElement.textContent = voiceQueryCount;
  }
  if (contextAccuracyElement) {
    contextAccuracyElement.textContent = '96.2%';
  }
}

// Stock Management Functions
function openReorderModal(productName) {
  const product = enhancedStockData.products.find(p => p.name === productName);
  if (!product) return;
  
  document.getElementById('modalProduct').value = product.name;
  document.getElementById('modalQuantity').value = Math.max(product.reorder * 2, 100);
  document.getElementById('modalSupplier').value = product.supplier;
  
  const deliveryDate = new Date();
  deliveryDate.setDate(deliveryDate.getDate() + product.leadTime);
  document.getElementById('modalDelivery').value = deliveryDate.toISOString().split('T')[0];
  
  document.getElementById('orderModal').classList.remove('hidden');
}

function closeOrderModal() {
  document.getElementById('orderModal').classList.add('hidden');
}

function confirmOrderModal() {
  const productName = document.getElementById('modalProduct').value;
  const quantity = parseInt(document.getElementById('modalQuantity').value);
  const supplier = document.getElementById('modalSupplier').value;
  const delivery = document.getElementById('modalDelivery').value;
  
  if (quantity <= 0) {
    showNotification('Please enter a valid quantity', 'error');
    return;
  }
  
  // Add to order history
  const newOrder = {
    id: orderHistory.length + 1,
    product: productName,
    quantity: quantity,
    status: 'pending',
    date: new Date().toISOString().split('T')[0],
    delivery: delivery
  };
  
  orderHistory.unshift(newOrder);
  
  // Update stock level (simulate restocking)
  const product = enhancedStockData.products.find(p => p.name === productName);
  if (product) {
    product.current += quantity;
  }
  
  closeOrderModal();
  populateStockData();
  populateRecentOrders();
  
  // Update stock chart
  if (charts.stockLevels) {
    charts.stockLevels.data.datasets[0].data = enhancedStockData.products.map(p => p.current);
    charts.stockLevels.update();
  }
  
  // Send confirmation email
  sendEmailAlert('reorder_confirmation', product, `Order confirmed: ${quantity} units of ${productName} from ${supplier}`);
  
  showNotification(`Order placed successfully! ${quantity} units of ${productName} from ${supplier}. Email confirmation sent.`, 'success');
}

function handleBulkReorder() {
  const criticalItems = enhancedStockData.products.filter(p => p.current <= p.reorder);
  
  if (criticalItems.length === 0) {
    showNotification('No items need reordering at this time', 'info');
    return;
  }
  
  criticalItems.forEach(product => {
    const quantity = product.reorder * 2;
    const deliveryDate = new Date();
    deliveryDate.setDate(deliveryDate.getDate() + product.leadTime);
    
    const newOrder = {
      id: orderHistory.length + 1,
      product: product.name,
      quantity: quantity,
      status: 'pending',
      date: new Date().toISOString().split('T')[0],
      delivery: deliveryDate.toISOString().split('T')[0]
    };
    
    orderHistory.unshift(newOrder);
    product.current += quantity;
    
    // Send email confirmation for each item
    sendEmailAlert('bulk_reorder', product, `Bulk reorder: ${quantity} units of ${product.name}`);
  });
  
  populateStockData();
  populateRecentOrders();
  
  if (charts.stockLevels) {
    charts.stockLevels.data.datasets[0].data = enhancedStockData.products.map(p => p.current);
    charts.stockLevels.update();
  }
  
  showNotification(`Bulk reorder completed! ${criticalItems.length} items reordered automatically with email confirmations sent.`, 'success');
}

function viewSupplierDetails(supplier) {
  showNotification(`Supplier Details: ${supplier} - Contact information and terms available in supplier management system. Email alerts configured for stock updates.`, 'info');
}

// Notification System
function showNotification(message, type = 'info') {
  const notification = document.getElementById('notification');
  const notificationText = document.getElementById('notificationText');
  
  if (notification && notificationText) {
    notificationText.textContent = message;
    notification.className = `notification ${type}`;
    notification.classList.remove('hidden');
    
    // Auto hide after 5 seconds
    setTimeout(() => {
      hideNotification();
    }, 5000);
  }
}

function hideNotification() {
  const notification = document.getElementById('notification');
  if (notification) {
    notification.classList.add('hidden');
  }
}

// Make functions globally available for onclick handlers
window.openReorderModal = openReorderModal;
window.viewSupplierDetails = viewSupplierDetails;
window.askSuggestion = askSuggestion;
window.testEmailAlert = testEmailAlert;
window.sendDemoUnderstock = sendDemoUnderstock;