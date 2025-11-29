/**
 * ML Classifier Metrics JavaScript
 * Handles display and updates for toxicity, safety, and escalation classifiers
 */

/**
 * Update classifier metrics display
 * Call this function after receiving analysis results from the API
 * 
 * @param {Object} data - Analysis result data from /api/analyze endpoint
 */
function updateClassifierMetrics(data) {
    // Show the metrics section
    document.getElementById('classifierMetrics').style.display = 'block';
    
    // Extract classifier data (adjust based on your actual API response structure)
    const toxicity = data.classifiers?.toxicity || data.toxicity_classifier || {};
    const safety = data.classifiers?.safety || data.safety_classifier || {};
    const escalation = data.classifiers?.escalation || data.escalation_detector || {};
    
    // Update Toxicity Metrics
    updateToxicityMetrics(toxicity);
    
    // Update Safety Metrics (Bio & Cyber)
    updateSafetyMetrics(safety);
    
    // Update Escalation Metrics
    updateEscalationMetrics(escalation);
    
    // Update Detailed Breakdown
    updateDetailedBreakdown(toxicity, safety, escalation);
}

function updateToxicityMetrics(toxicity) {
    const score = toxicity.score || 0;
    const severity = toxicity.severity || 'none';
    const flags = toxicity.flags || [];
    
    document.getElementById('toxicity-score').textContent = score.toFixed(2);
    document.getElementById('toxicity-severity').innerHTML = 
        `<i class="fas ${getSeverityIcon(severity)}"></i> <span>${capitalize(severity)}</span>`;
    document.getElementById('toxicity-severity').style.color = getSeverityColor(severity);
    document.getElementById('toxicity-flags').textContent = 
        flags.length > 0 ? flags.join(', ') : 'No flags';
}

function updateSafetyMetrics(safety) {
    const bioScore = safety.bio_safety_score || 0;
    const cyberScore = safety.cyber_safety_score || 0;
    const bioSeverity = safety.bio_severity || 'none';
    const cyberSeverity = safety.cyber_severity || 'none';
    const bioFlags = safety.bio_flags || [];
    const cyberFlags = safety.cyber_flags || [];
    
    // Bio-Safety
    document.getElementById('bio-safety-score').textContent = bioScore.toFixed(2);
    document.getElementById('bio-severity').innerHTML = 
        `<i class="fas ${getSeverityIcon(bioSeverity)}"></i> <span>${capitalize(bioSeverity)}</span>`;
    document.getElementById('bio-severity').style.color = getSeverityColor(bioSeverity);
    document.getElementById('bio-flags').textContent = 
        bioFlags.length > 0 ? bioFlags.join(', ') : 'No flags';
    
    // Cyber-Safety
    document.getElementById('cyber-safety-score').textContent = cyberScore.toFixed(2);
    document.getElementById('cyber-severity').innerHTML = 
        `<i class="fas ${getSeverityIcon(cyberSeverity)}"></i> <span>${capitalize(cyberSeverity)}</span>`;
    document.getElementById('cyber-severity').style.color = getSeverityColor(cyberSeverity);
    document.getElementById('cyber-flags').textContent = 
        cyberFlags.length > 0 ? cyberFlags.join(', ') : 'No flags';
}

function updateEscalationMetrics(escalation) {
    const score = escalation.escalation_score || 0;
    const severity = escalation.severity || 'none';
    const trend = escalation.trend || 'stable';
    const flags = escalation.flags || [];
    
    document.getElementById('escalation-score').textContent = score.toFixed(2);
    document.getElementById('escalation-trend').innerHTML = 
        `<i class="fas ${getTrendIcon(trend)}"></i> <span>${capitalize(trend)}</span>`;
    document.getElementById('escalation-trend').style.color = getTrendColor(trend);
    document.getElementById('escalation-flags').textContent = 
        flags.length > 0 ? flags.join(', ') : 'No flags';
}

function updateDetailedBreakdown(toxicity, safety, escalation) {
    const details = toxicity.details || {};
    const safetyDetails = safety.details || {};
    const escalationDetails = escalation.details || {};
    
    let html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">';
    
    // Toxicity Details
    html += `
        <div>
            <h4 style="color: #ff3232; margin-bottom: 0.5rem; font-size: 1rem;">
                <i class="fas fa-biohazard"></i> Toxicity Breakdown
            </h4>
            <p><strong>Keyword Score:</strong> ${(details.keyword_score || 0).toFixed(3)}</p>
            <p><strong>Pattern Score:</strong> ${(details.pattern_score || 0).toFixed(3)}</p>
            <p><strong>Profanity Score:</strong> ${(details.profanity_score || 0).toFixed(3)}</p>
            <p><strong>Slur Score:</strong> ${(details.slur_score || 0).toFixed(3)}</p>
        </div>
    `;
    
    // Safety Details
    html += `
        <div>
            <h4 style="color: #ffa500; margin-bottom: 0.5rem; font-size: 1rem;">
                <i class="fas fa-virus"></i> Bio-Safety Breakdown
            </h4>
            <p><strong>Keyword Score:</strong> ${(safetyDetails.bio_keyword_score || 0).toFixed(3)}</p>
            <p><strong>Pattern Score:</strong> ${(safetyDetails.bio_pattern_score || 0).toFixed(3)}</p>
            <p><strong>Overall Risk:</strong> ${capitalize(safety.overall_risk || 'low')}</p>
        </div>
    `;
    
    html += `
        <div>
            <h4 style="color: #64c8ff; margin-bottom: 0.5rem; font-size: 1rem;">
                <i class="fas fa-shield-virus"></i> Cyber-Safety Breakdown
            </h4>
            <p><strong>Keyword Score:</strong> ${(safetyDetails.cyber_keyword_score || 0).toFixed(3)}</p>
            <p><strong>Pattern Score:</strong> ${(safetyDetails.cyber_pattern_score || 0).toFixed(3)}</p>
            <p><strong>Overall Risk:</strong> ${capitalize(safety.overall_risk || 'low')}</p>
        </div>
    `;
    
    // Escalation Details
    html += `
        <div>
            <h4 style="color: var(--primary-lime); margin-bottom: 0.5rem; font-size: 1rem;">
                <i class="fas fa-chart-line"></i> Escalation Breakdown
            </h4>
            <p><strong>Current Message:</strong> ${(escalationDetails.current_message_score || 0).toFixed(3)}</p>
            <p><strong>History Score:</strong> ${(escalationDetails.history_score || 0).toFixed(3)}</p>
            <p><strong>Message Count:</strong> ${escalationDetails.message_count || 0}</p>
            <p><strong>Trend:</strong> ${capitalize(escalation.trend || 'stable')}</p>
        </div>
    `;
    
    html += '</div>';
    
    document.getElementById('detailedBreakdown').innerHTML = html;
}

// Helper Functions
function getSeverityIcon(severity) {
    const icons = {
        'critical': 'fa-exclamation-circle',
        'high': 'fa-exclamation-triangle',
        'medium': 'fa-exclamation',
        'low': 'fa-info-circle',
        'none': 'fa-check-circle'
    };
    return icons[severity] || 'fa-info-circle';
}

function getSeverityColor(severity) {
    const colors = {
        'critical': '#ff0000',
        'high': '#ff6600',
        'medium': '#ffa500',
        'low': '#ffcc00',
        'none': '#00ff00'
    };
    return colors[severity] || '#888';
}

function getTrendIcon(trend) {
    const icons = {
        'escalating': 'fa-arrow-up',
        'de-escalating': 'fa-arrow-down',
        'stable': 'fa-minus',
        'insufficient_data': 'fa-question-circle'
    };
    return icons[trend] || 'fa-minus';
}

function getTrendColor(trend) {
    const colors = {
        'escalating': '#ff0000',
        'de-escalating': '#00ff00',
        'stable': 'var(--primary-lime)',
        'insufficient_data': '#888'
    };
    return colors[trend] || '#888';
}

function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1).replace('_', ' ');
}
