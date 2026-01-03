/**
 * shows the cards carousel
 * @param {Array} cardsData cards data
 */
function showCardsCarousel(cardsData) {
    let cardsCarousel = `<div class="cards" id="paginated_cards">`;
   
    cardsData.forEach((card, index) => {
      cardsCarousel += `
        <div class="card">
          <img src="${card.image}" alt="${card.title || card.name}">
          <div class="card-content">
            <div class="card-title">${card.title || card.name}</div>
            ${card.ratings ? `<div class="card-text">⭐ ${card.ratings}</div>` : ''}
            ${card.description ? `<div class="card-text">${card.description}</div>` : ''}
          </div>
        </div>
      `;
    });
   
    cardsCarousel += `</div>`;
   
    $(cardsCarousel).appendTo(".chats");
    scrollToBottomOfResults();
  }