/**
 * Adaptive Card Components - Custom renderers for specific card types
 * 
 * This module provides helper functions for rendering adaptive cards
 * with proper styling and interaction handling.
 */

/**
 * Renders a leave request form card
 * @param {Object} cardData - The adaptive card JSON payload
 * @param {Object} metadata - Additional metadata like employee_id
 */
function renderLeaveRequestCard(cardData, metadata = {}) {
    console.log("📋 Rendering Leave Request Card");
    console.log("Card data:", cardData);
    console.log("Metadata:", metadata);
    
    // Use the main adaptive card renderer
    renderAdaptiveCardPayload({
        data: {
            card: cardData,
            metadata: {
                ...metadata,
                template: "leave_request"
            }
        }
    });
}

/**
 * Renders a leave validation response card
 * @param {Object} cardData - The adaptive card JSON payload
 * @param {Object} metadata - Additional metadata
 */
function renderLeaveResponseCard(cardData, metadata = {}) {
    console.log("✅ Rendering Leave Response Card");
    console.log("Card data:", cardData);
    console.log("Metadata:", metadata);
    
    // Use the main adaptive card renderer
    renderAdaptiveCardPayload({
        data: {
            card: cardData,
            metadata: {
                ...metadata,
                template: "leave_response"
            }
        }
    });
}

/**
 * Renders a contact form card
 * @param {Object} cardData - The adaptive card JSON payload
 * @param {Object} metadata - Additional metadata
 */
function renderContactFormCard(cardData, metadata = {}) {
    console.log("📝 Rendering Contact Form Card");
    
    renderAdaptiveCardPayload({
        data: {
            card: cardData,
            metadata: {
                ...metadata,
                template: "contact_form"
            }
        }
    });
}

/**
 * Creates a simple leave request form card programmatically
 * @param {Object} options - Form options
 * @param {string} options.startDate - Pre-filled start date
 * @param {string} options.endDate - Pre-filled end date
 * @param {string} options.leaveType - Pre-filled leave type
 * @param {string} options.employeeId - Employee ID
 */
function createLeaveRequestCard(options = {}) {
    const card = {
        "type": "AdaptiveCard",
        "version": "1.5",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "body": [
            {
                "type": "TextBlock",
                "text": "📅 Leave Request",
                "weight": "Bolder",
                "size": "Large",
                "spacing": "None"
            },
            {
                "type": "TextBlock",
                "text": "Submit your leave request below",
                "wrap": true,
                "size": "Small",
                "isSubtle": true,
                "spacing": "Small"
            },
            {
                "type": "Container",
                "spacing": "Medium",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": "**Start Date**",
                        "spacing": "Small"
                    },
                    {
                        "type": "Input.Date",
                        "id": "start_date",
                        "placeholder": "Select start date",
                        "value": options.startDate || ""
                    },
                    {
                        "type": "TextBlock",
                        "text": "**End Date**",
                        "spacing": "Medium"
                    },
                    {
                        "type": "Input.Date",
                        "id": "end_date",
                        "placeholder": "Select end date",
                        "value": options.endDate || ""
                    },
                    {
                        "type": "TextBlock",
                        "text": "**Leave Type**",
                        "spacing": "Medium"
                    },
                    {
                        "type": "Input.ChoiceSet",
                        "id": "leave_type",
                        "style": "compact",
                        "placeholder": "Select leave type",
                        "value": options.leaveType || "",
                        "choices": [
                            {"title": "Sick Leave", "value": "sick leave"},
                            {"title": "Annual Leave", "value": "annual leave"},
                            {"title": "Flexi Leave", "value": "flexi leave"},
                            {"title": "Unpaid Leave", "value": "unpaid leave"}
                        ]
                    }
                ]
            }
        ],
        "actions": [
            {
                "type": "Action.Submit",
                "title": "✅ Submit Request",
                "style": "positive",
                "data": {
                    "submit": "leave_calculator_submit",
                    "intent": "submit_leave_form"
                }
            }
        ]
    };
    
    return card;
}

/**
 * Creates a leave validation result card
 * @param {Object} data - Validation data from API
 * @param {boolean} data.eligible - Whether leave is approved
 * @param {string} data.message - Status message
 * @param {number} data.requested_days - Days requested
 * @param {number} data.remaining_days - Balance after leave
 * @param {string} data.leave_type - Type of leave
 * @param {string} data.start_date - Leave start date
 * @param {string} data.end_date - Leave end date
 */
function createLeaveValidationCard(data) {
    const statusColor = data.eligible ? "Good" : "Attention";
    const statusIcon = data.eligible ? "✅" : "⚠️";
    
    const card = {
        "type": "AdaptiveCard",
        "version": "1.5",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "body": [
            {
                "type": "Container",
                "style": data.eligible ? "good" : "attention",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": `${statusIcon} ${data.message || 'Leave Request Status'}`,
                        "weight": "Bolder",
                        "size": "Medium",
                        "wrap": true,
                        "color": statusColor
                    }
                ]
            },
            {
                "type": "FactSet",
                "spacing": "Medium",
                "facts": [
                    {
                        "title": "📅 Dates",
                        "value": `${data.start_date} to ${data.end_date}`
                    },
                    {
                        "title": "📋 Type",
                        "value": (data.leave_type || "").replace(/\b\w/g, l => l.toUpperCase())
                    },
                    {
                        "title": "📊 Days Requested",
                        "value": `${data.requested_days || 0} day(s)`
                    },
                    {
                        "title": "💰 Balance After",
                        "value": `${data.remaining_days || 0} day(s)`
                    }
                ]
            }
        ]
    };
    
    // Add current balance section if available
    if (data.sick_leave_days !== undefined || data.vacation_days !== undefined) {
        card.body.push({
            "type": "Container",
            "spacing": "Medium",
            "items": [
                {
                    "type": "TextBlock",
                    "text": "**Current Leave Balance**",
                    "weight": "Bolder"
                },
                {
                    "type": "ColumnSet",
                    "columns": [
                        {
                            "type": "Column",
                            "width": "stretch",
                            "items": [
                                {
                                    "type": "TextBlock",
                                    "text": `🏥 Sick: ${data.sick_leave_days || 0} days`,
                                    "horizontalAlignment": "Center"
                                }
                            ]
                        },
                        {
                            "type": "Column",
                            "width": "stretch",
                            "items": [
                                {
                                    "type": "TextBlock",
                                    "text": `🏖️ Vacation: ${data.vacation_days || 0} days`,
                                    "horizontalAlignment": "Center"
                                }
                            ]
                        }
                    ]
                }
            ]
        });
    }
    
    return card;
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        renderLeaveRequestCard,
        renderLeaveResponseCard,
        renderContactFormCard,
        createLeaveRequestCard,
        createLeaveValidationCard
    };
}
