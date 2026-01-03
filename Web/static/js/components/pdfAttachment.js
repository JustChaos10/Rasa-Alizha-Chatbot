/**
 * renders the pdf attachment
 * @param {Object} response pdf attachment response
 */
function renderPdfAttachment(response) {
  const { title, url } = response.custom;
 
  const pdfAttachment = `
    <div class="pdf_attachment" onclick="window.open('${url}', '_blank')">
      <i class="fa fa-file-pdf-o" style="color: #dc3545; font-size: 24px;"></i>
      <div class="pdf-info">
        <div class="pdf-title">${title}</div>
        <div class="pdf-size">Click to open PDF</div>
      </div>
      <i class="fa fa-download" style="color: #6c757d;"></i>
    </div>
  `;
 
  $(pdfAttachment).appendTo(".chats");
  scrollToBottomOfResults();
}
