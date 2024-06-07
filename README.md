# Morris

### **Note: Still in development**

Morris is a real-time voice assistant that seeks to remember much more than typical voice assistants through the following improvements: 
1. Always-on functionality: Morris is designed to run in the background as a distinct program, listening and categorizing everything. 
2. Provide near-instantaneous responses
3. Run on device

These improvements enable Morris to effectively remember everything it ever hears and eliminate the requirement that voice assistants are only useful and relevant when they are explicitly called.

To do this, we construct the voice assistant as a "layered" inference model, whereby distinct instances of the voice transcription are staggered at periodic intervals, while inference is done in a batch manner for all layers. Since voice transcription improves with context, we would like the slowest layer to provide the most accurate transcription, and subsequent earlier layers to be less accurate but transcribed faster to retain quick response latency. 
