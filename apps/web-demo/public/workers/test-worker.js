// Simple test worker to verify message passing
self.postMessage({
  type: 'info',
  message: 'Test worker starting'
});

self.onmessage = (event) => {
  self.postMessage({
    type: 'info',
    message: 'Test worker received: ' + JSON.stringify(event.data)
  });
  
  if (event.data.type === 'init') {
    self.postMessage({
      type: 'info',
      message: 'Test worker sending ready'
    });
    
    // Send the ready status
    self.postMessage({
      type: 'status',
      status: 'ready',
      message: 'Test ready!',
      voices: {}
    });
  }
};

self.postMessage({
  type: 'info',
  message: 'Test worker ready to receive messages'
});