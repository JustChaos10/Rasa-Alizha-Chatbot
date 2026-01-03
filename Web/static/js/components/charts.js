let chatChart;
let modalChart;
 
/**
 * creates the chart
 * @param {String} title chart title
 * @param {Array} labels chart labels
 * @param {Array} backgroundColor chart background colors
 * @param {Array} chartsData chart data
 * @param {String} chartType chart type
 * @param {String} displayLegend display legend
 */
function createChart(title, labels, backgroundColor, chartsData, chartType, displayLegend) {
  const chartContainer = `
    <div class="chart-container">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <h6 style="margin: 0;">${title}</h6>
        <button id="expand" style="background: none; border: none; cursor: pointer; font-size: 16px;">⛶</button>
      </div>
      <canvas id="chatChart" width="300" height="200"></canvas>
    </div>
  `;
 
  $(chartContainer).appendTo(".chats");
 
  const ctx = document.getElementById("chatChart").getContext("2d");
 
  // Destroy existing chart if it exists
  if (chatChart) {
    chatChart.destroy();
  }
 
  chatChart = new Chart(ctx, {
    type: chartType,
    data: {
      labels: labels,
      datasets: [{
        data: chartsData,
        backgroundColor: backgroundColor,
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: displayLegend === "true"
        }
      }
    }
  });
 
  scrollToBottomOfResults();
}
 
/**
 * creates the chart in modal
 * @param {String} title chart title
 * @param {Array} labels chart labels
 * @param {Array} backgroundColor chart background colors
 * @param {Array} chartsData chart data
 * @param {String} chartType chart type
 * @param {String} displayLegend display legend
 */
function createChartinModal(title, labels, backgroundColor, chartsData, chartType, displayLegend) {
  // Create modal if it doesn't exist
  if (!$("#chartModal").length) {
    const modal = `
      <div id="chartModal" class="modal">
        <div class="modal-content">
          <span class="close" style="float: right; font-size: 28px; font-weight: bold; cursor: pointer;">&times;</span>
          <h4>${title}</h4>
          <canvas id="modal-chart" width="400" height="300"></canvas>
        </div>
      </div>
    `;
    $("body").append(modal);
   
    // Close modal functionality
    $(".close, #chartModal").on("click", function(e) {
      if (e.target === this) {
        $("#chartModal").hide();
        if (modalChart) {
          modalChart.destroy();
        }
      }
    });
  }
 
  $("#chartModal").show();
 
  const modalCtx = document.getElementById("modal-chart").getContext("2d");
 
  // Destroy existing modal chart if it exists
  if (modalChart) {
    modalChart.destroy();
  }
 
  modalChart = new Chart(modalCtx, {
    type: chartType,
    data: {
      labels: labels,
      datasets: [{
        data: chartsData,
        backgroundColor: backgroundColor,
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: displayLegend === "true"
        }
      }
    }
  });
}