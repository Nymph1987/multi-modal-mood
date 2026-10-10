/**
 * Main JavaScript for Emotion Recognition System
 * Handles polling for current emotion status and history logs.
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log("System initialized.");

    // Polling interval in milliseconds
    const POLL_INTERVAL = 200;
    const HISTORY_FALLBACK_INTERVAL = 5000;
    const baseUrl = (window.location && window.location.origin && window.location.origin !== "null")
        ? window.location.origin
        : "http://127.0.0.1:5000";

    // DOM Elements
    const currentEmotionEl = document.getElementById('current-emotion');
    const currentConfidenceEl = document.getElementById('current-confidence');
    const logTableBody = document.querySelector('#log-table tbody');
    const videoStreamEl = document.getElementById('video-stream');
    const systemStatusEl = document.getElementById('system-status');
    let lastEmotion = null;
    let lastConfidence = null;
    let lastTimestamp = 0;
    let lastLogTs = 0;
    let lastHistoryFetchAt = 0;
    let statusInFlight = false;
    let historyInFlight = false;

    if (videoStreamEl) {
        videoStreamEl.src = `${baseUrl}/video_feed?t=${Date.now()}`;
        videoStreamEl.onerror = () => {
            if (systemStatusEl) {
                systemStatusEl.textContent = `Video Feed Error (${baseUrl})`;
                systemStatusEl.className = 'status-error';
            }
        };
        videoStreamEl.onload = () => {
            if (systemStatusEl) {
                systemStatusEl.textContent = 'Running';
                systemStatusEl.className = 'status-ok';
            }
        };
    }

    /**
     * Update the current emotion display.
     */
    function updateCurrentStatus() {
        if (statusInFlight) {
            return;
        }
        statusInFlight = true;
        // Removed 'since' parameter to force always getting data
        fetch(`${baseUrl}/api/current_emotion?t=${Date.now()}`, { cache: 'no-store', mode: 'cors', credentials: 'omit' })
            .then(response => {
                if (response.status === 204) {
                    return null;
                }
                return response.json();
            })
            .then(data => {
                if (!data || !data.emotion) {
                    return;
                }

                const emotionChanged = (lastEmotion === null) || (data.emotion !== lastEmotion);
                currentEmotionEl.textContent = data.emotion;
                const confValue = Number(data.confidence);
                let confidenceChanged = false;
                if (Number.isFinite(confValue)) {
                    currentConfidenceEl.textContent = (confValue * 100).toFixed(1) + "%";
                    if (typeof lastConfidence === 'number') {
                        confidenceChanged = Math.abs(confValue - lastConfidence) >= 1e-4;
                    } else {
                        confidenceChanged = true;
                    }
                    lastConfidence = confValue;
                } else {
                    currentConfidenceEl.textContent = String(data.confidence ?? "");
                    const confText = String(data.confidence ?? "");
                    confidenceChanged = (lastConfidence === null) || (confText !== String(lastConfidence));
                    lastConfidence = confText;
                }
                lastEmotion = data.emotion;

                if (typeof data.timestamp === 'number' && data.timestamp > lastTimestamp) {
                    lastTimestamp = data.timestamp;
                }

                const now = Date.now();
                let logChanged = false;
                if (typeof data.last_log_ts === 'number' && data.last_log_ts > lastLogTs) {
                    lastLogTs = data.last_log_ts;
                    logChanged = true;
                }

                if (logChanged || (now - lastHistoryFetchAt) >= HISTORY_FALLBACK_INTERVAL) {
                    updateHistory();
                    lastHistoryFetchAt = now;
                }
            })
            .catch(err => console.error("Error fetching status:", err))
            .finally(() => {
                statusInFlight = false;
            });
    }

    /**
     * Update the history log table.
     */
    function updateHistory() {
        if (historyInFlight) {
            return;
        }
        historyInFlight = true;
        fetch(`${baseUrl}/api/history?t=${Date.now()}`, { cache: 'no-store', mode: 'cors', credentials: 'omit' })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                return response.json();
            })
            .then(logs => {
                // Clear existing table content
                logTableBody.innerHTML = '';

                if (logs.length === 0) {
                    const row = document.createElement('tr');
                    row.innerHTML = '<td colspan="3">No logs available yet.</td>';
                    logTableBody.appendChild(row);
                    return;
                }

                logs.forEach(log => {
                    const row = document.createElement('tr');
                    // Ensure keys match the CSV header or dict returned by backend
                    // Backend dict keys might be lowercase from DictReader or LogManager
                    // CSV header: Timestamp, DateTime, Emotion, Confidence
                    // DictReader uses header keys.
                    
                    const timeStr = log['DateTime'] || log['timestamp'] || '-';
                    const emotion = log['Emotion'] || log['emotion'] || 'Unknown';
                    const confidence = log['Confidence'] || log['confidence'] || '0';

                    row.innerHTML = `
                        <td>${timeStr}</td>
                        <td>${emotion}</td>
                        <td>${confidence}</td>
                    `;
                    logTableBody.appendChild(row);
                });
            })
            .catch(err => {
                logTableBody.innerHTML = '';
                const row = document.createElement('tr');
                row.innerHTML = `<td colspan="3">Failed to load logs: ${String(err)}</td>`;
                logTableBody.appendChild(row);
            })
            .finally(() => {
                historyInFlight = false;
            });
    }

    // Start polling
    setInterval(() => {
        updateCurrentStatus();
    }, POLL_INTERVAL);
    
    updateHistory();
    updateCurrentStatus();
});
