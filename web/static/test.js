// Basic initialization check
console.log("Script loaded!");

// Booking Flow Functions
function startBookingFlow() {
    console.log("Booking flow started!");
    // Temporary test implementation
    const bookingId = Math.floor(100000 + Math.random() * 900000);
    showBookingConfirmation({
        booking_id: bookingId,
        details: {
            sender: "Test Sender",
            receiver: "Test Receiver",
            package_details: "Test Package",
            weight: "5kg",
            dimensions: "30x20x15cm",
            is_return: false,
            address_notes: "Test notes"
        }
    });
}

// Basic UI Functions
function showBookingConfirmation(data) {
    console.log("Showing booking:", data);
    const section = document.getElementById('booking-section');
    section.classList.remove('hidden');
    
    document.getElementById('booking-id').textContent = data.booking_id;
    document.getElementById('booking-details').innerHTML = `
        <div class="space-y-2">
            <p><strong>Sender:</strong> ${data.details.sender}</p>
            <p><strong>Receiver:</strong> ${data.details.receiver}</p>
            <p><strong>Package:</strong> ${data.details.package_details}</p>
            <p><strong>Weight:</strong> ${data.details.weight}</p>
            <p><strong>Dimensions:</strong> ${data.details.dimensions}</p>
            <p><strong>Return:</strong> ${data.details.is_return ? 'Yes' : 'No'}</p>
            <p><strong>Notes:</strong> ${data.details.address_notes}</p>
        </div>
    `;
}

function hideBooking() {
    document.getElementById('booking-section').classList.add('hidden');
}

// Initial setup
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded");
    document.getElementById('create-booking-btn').addEventListener('click', startBookingFlow);
});