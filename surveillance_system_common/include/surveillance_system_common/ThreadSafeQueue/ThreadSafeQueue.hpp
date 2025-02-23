#ifndef THREAD_SAFE_QUEUE_HPP
#define THREAD_SAFE_QUEUE_HPP

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

template <typename T>
class ThreadSafeQueue {
    public:
        ThreadSafeQueue() : queueMaxSize(0), stopRequested(false), isInitialized(false) {}

        explicit ThreadSafeQueue(size_t maxSize) : queueMaxSize(0), stopRequested(false), isInitialized(false) {
            initialize(maxSize);
        }

        ThreadSafeQueue(const ThreadSafeQueue&) = delete;
        ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;
        ThreadSafeQueue(ThreadSafeQueue&&) = delete;
        ThreadSafeQueue& operator=(ThreadSafeQueue&&) = delete;

        bool initialize(size_t maxSize) {
            std::lock_guard<std::mutex> lock(queueMutex);
            if (isInitialized) {
                return false;
            }
            queueMaxSize = maxSize;
            isInitialized = true;
            return true;
        }

        bool push(T item) {
            std::unique_lock<std::mutex> lock(queueMutex);  
            if(!isInitialized) return false;

            queueProducer.wait(lock, [this] {  
                return stopRequested || (queueMaxSize == 0 || internalQueue.size() < queueMaxSize); 
            });
            if (stopRequested) { 
                return false;
            }
            internalQueue.push(std::move(item));  
            lock.unlock();                        
            queueConsumer.notify_one();          
            return true;
        }

        bool try_push(T item) {
            if(!isInitialized) return false;
            std::unique_lock<std::mutex> lock(queueMutex);
            if (!isInitialized ||stopRequested || (queueMaxSize > 0 && internalQueue.size() >= queueMaxSize)) {
                return false;
            }
            internalQueue.push(std::move(item));
            lock.unlock();

            queueConsumer.notify_one();
            return true;
        }

        std::optional<T> pop() {
            if(!isInitialized) return std::nullopt;
            std::unique_lock<std::mutex> lock(queueMutex); 
            queueConsumer.wait(lock, [this] {                    
                return stopRequested || !internalQueue.empty();  
            });
            if (stopRequested && internalQueue.empty()) {  
                return std::nullopt;                       
            }
            T item = std::move(internalQueue.front());  
            internalQueue.pop();                        
            lock.unlock();                              
            queueProducer.notify_one();                 
            return item;                                
        }

        bool isQueueEmpty() const {
            if(!isInitialized) return false;
            std::lock_guard<std::mutex> lock(queueMutex);
            return internalQueue.empty();
        }

        size_t getQueueSize() const {
            if(!isInitialized) return 0;
            std::lock_guard<std::mutex> lock(queueMutex);
            return internalQueue.size();
        }

        void stopWaitingThreads() {
            if(!isInitialized) return;
            std::lock_guard<std::mutex> lock(queueMutex);
            stopRequested = true;

            queueConsumer.notify_all();
            queueProducer.notify_all();
        }

    private:
        std::queue<T> internalQueue;
        mutable std::mutex queueMutex;
        std::condition_variable queueProducer;
        std::condition_variable queueConsumer;
        size_t queueMaxSize; 
        bool stopRequested;
        bool isInitialized;
};

#endif // THREAD_SAFE_QUEUE_HPP

