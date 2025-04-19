#ifndef THREAD_SAFE_QUEUE_HPP
#define THREAD_SAFE_QUEUE_HPP

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

template <typename T>
class ThreadSafeQueue {
    public:
        explicit ThreadSafeQueue(size_t maxSize = 0) : queueMaxSize(maxSize), stopRequested(false) {}

        ThreadSafeQueue(const ThreadSafeQueue&) = delete;
        ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;
        ThreadSafeQueue(ThreadSafeQueue&&) = delete;
        ThreadSafeQueue& operator=(ThreadSafeQueue&&) = delete;

        bool push(T item) {
            std::unique_lock<std::mutex> lock(queueMutex);  
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
            std::unique_lock<std::mutex> lock(queueMutex);
            if (stopRequested || (queueMaxSize > 0 && internalQueue.size() >= queueMaxSize)) {
                return false;
            }
            internalQueue.push(std::move(item));
            lock.unlock();

            queueConsumer.notify_one();
            return true;
        }

        std::optional<T> pop() {
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
            std::lock_guard<std::mutex> lock(queueMutex);
            return internalQueue.empty();
        }

        size_t getQueueSize() const {
            std::lock_guard<std::mutex> lock(queueMutex);
            return internalQueue.size();
        }

        void stopWaitingThreads() {
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
};

#endif // THREAD_SAFE_QUEUE_HPP

