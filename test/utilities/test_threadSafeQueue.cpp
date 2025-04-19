#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <numeric>
#include <atomic>
#include <chrono>
#include <string>
#include <future> 

#include "ThreadSafeQueue.hpp"

class ThreadSafeQueueTests : public ::testing::Test {
    protected:
        void SetUp() override {}
        void TearDown() override {}
};

TEST_F(ThreadSafeQueueTests, InitialStateIsEmpty) {
    ThreadSafeQueue<int> q;
    ASSERT_TRUE(q.isQueueEmpty());
    ASSERT_EQ(q.getQueueSize(), 0);
}

TEST_F(ThreadSafeQueueTests, PushPopSingleElement) {
    ThreadSafeQueue<int> q;
    ASSERT_TRUE(q.push(10));
    ASSERT_FALSE(q.isQueueEmpty());
    ASSERT_EQ(q.getQueueSize(), 1);

    std::optional<int> val = q.pop();
    ASSERT_TRUE(val.has_value());
    ASSERT_EQ(val.value(), 10);
    ASSERT_TRUE(q.isQueueEmpty());
    ASSERT_EQ(q.getQueueSize(), 0);
}

TEST_F(ThreadSafeQueueTests, TryPushPopSingleElement) {
    ThreadSafeQueue<std::string> q;
    std::string test_str = "hello";
    ASSERT_TRUE(q.try_push(test_str));
    ASSERT_FALSE(q.isQueueEmpty());
    ASSERT_EQ(q.getQueueSize(), 1);

    std::optional<std::string> val = q.pop();
    ASSERT_TRUE(val.has_value());
    ASSERT_EQ(val.value(), test_str);
    ASSERT_TRUE(q.isQueueEmpty());
    ASSERT_EQ(q.getQueueSize(), 0);
}

TEST_F(ThreadSafeQueueTests, PushMultiplePopMultiple) {
    ThreadSafeQueue<int> q;
    const int count = 5;
    for (int i = 0; i < count; ++i) {
        ASSERT_TRUE(q.push(i));
    }
    ASSERT_EQ(q.getQueueSize(), count);

    for (int i = 0; i < count; ++i) {
        std::optional<int> val = q.pop();
        ASSERT_TRUE(val.has_value());
        ASSERT_EQ(val.value(), i);
    }
    ASSERT_TRUE(q.isQueueEmpty());
}


TEST_F(ThreadSafeQueueTests, BoundedQueueTryPushFull) {
    const size_t maxSize = 2;
    ThreadSafeQueue<int> q(maxSize);

    ASSERT_TRUE(q.try_push(1));
    ASSERT_TRUE(q.try_push(2));
    ASSERT_EQ(q.getQueueSize(), maxSize);

    /*
        The queue should be full - the push should fail
    */
    ASSERT_FALSE(q.try_push(3));
    ASSERT_EQ(q.getQueueSize(), maxSize);

    
    auto val = q.pop();
    ASSERT_TRUE(val.has_value());
    ASSERT_EQ(val.value(), 1);
    ASSERT_EQ(q.getQueueSize(), maxSize - 1);

    /*
        After popping one elment, the push should succeed
    */
    ASSERT_TRUE(q.try_push(4));
    ASSERT_EQ(q.getQueueSize(), maxSize);
}


TEST_F(ThreadSafeQueueTests, StopEmptyQueuePopReturnsNullopt) {
    ThreadSafeQueue<int> q;
    ASSERT_TRUE(q.isQueueEmpty());

    auto future = std::async(std::launch::async, [&]() {
        return q.pop(); // block execution
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    q.stopWaitingThreads();

    std::optional<int> result = future.get();

    ASSERT_FALSE(result.has_value());
}

TEST_F(ThreadSafeQueueTests, QueueStopAllowsRemainingItemsToBePopped) {
    ThreadSafeQueue<int> q;
    q.push(1);
    q.push(2);
    ASSERT_EQ(q.getQueueSize(), 2);

    q.stopWaitingThreads();

    auto val1 = q.pop();
    ASSERT_TRUE(val1.has_value());
    ASSERT_EQ(val1.value(), 1);
    ASSERT_EQ(q.getQueueSize(), 1);

    auto val2 = q.pop();
    ASSERT_TRUE(val2.has_value());
    ASSERT_EQ(val2.value(), 2);
    ASSERT_TRUE(q.isQueueEmpty());

    auto val3 = q.pop();
    ASSERT_FALSE(val3.has_value());
}

TEST_F(ThreadSafeQueueTests, SingleProducerSingleConsumer) {
    ThreadSafeQueue<int> q;
    const int numItems = 1000;
    std::atomic<int> poppedItems = 0;
    std::vector<int> poppedValues;
    poppedValues.reserve(numItems);
    std::mutex vectorMtx;

    std::thread producer([&]() {
        for (int i = 0; i < numItems; ++i) {
            ASSERT_TRUE(q.push(i));
        }
    });

    std::thread consumer([&]() {
        int poppedCounter = 0;
        while (poppedCounter  < numItems) {
            std::optional<int> val = q.pop();
            if (val.has_value()) {
                std::lock_guard<std::mutex> lock(vectorMtx);
                poppedValues.push_back(val.value());
                poppedItems++;
                poppedCounter ++;
            } else {
                if (poppedItems >= numItems)
                    break;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    });

    producer.join();
    consumer.join();

    ASSERT_EQ(poppedItems.load(), numItems);
    ASSERT_EQ(poppedValues.size(),numItems);

    std::sort( poppedValues.begin(), poppedValues.end());
    bool correct = true;
    for (int i = 0; i < numItems; ++i) {
        if ( poppedValues[i] != i) {
            correct = false;
            break;
        }
    }
    ASSERT_TRUE(correct) << "Popped values do not match expected sequence.";
}

TEST_F(ThreadSafeQueueTests, MultipleProducersMultipleConsumers) {
    ThreadSafeQueue<int> q(100);
    const int producerCount = 4;
    const int consumerCount = 3;
    const int items_per_producer = 500;
    const int total_items = producerCount * items_per_producer;
    std::atomic<int> producedItems = 0;
    std::atomic<int> consumedItems = 0;

    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;

    consumers.reserve(consumerCount);
    producers.reserve(producerCount);

    for (int i = 0; i < producerCount; ++i) {
        producers.emplace_back([&, producerIdentifier = i]() {
            for (int j = 0; j < items_per_producer; ++j) {
                int value = producerIdentifier * items_per_producer + j;
                ASSERT_TRUE(q.push(value));
                producedItems++;
            }
        });
    }

    for (int i = 0; i < consumerCount; ++i) {
        consumers.emplace_back([&](){
            while(true) {
                std::optional<int> val = q.pop();
                if(val.has_value()) {
                    ++consumedItems;
                } else {
                    break;
                }
            }
        });
    }

    for (auto& t : producers) {
        t.join();
    }
    /* Sanity check */
    ASSERT_EQ(producedItems.load(), total_items);
    q.stopWaitingThreads();

    /*
        All consumer threads have to finish and eventually exit the loop, i.e. all the threads have
       to join
    */
    for (auto& t : consumers) {
        t.join();
    }
    ASSERT_EQ(consumedItems.load(), total_items);
    ASSERT_TRUE(q.isQueueEmpty());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
