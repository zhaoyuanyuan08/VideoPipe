#pragma once

#include <condition_variable>
#include <chrono>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace vp_sdk {

    template <typename T>
    class vp_sdk_blocking_queue {
    private:
        std::deque<T> items;
        mutable std::mutex lock;
        std::condition_variable ready;
        std::size_t capacity;
        bool closed = false;
        std::size_t dropped = 0;

    public:
        explicit vp_sdk_blocking_queue(std::size_t capacity): capacity(capacity == 0 ? 1 : capacity) {
        }

        void push_latest(T item) {
            {
                std::lock_guard<std::mutex> guard(lock);
                if (closed) {
                    return;
                }
                while (items.size() >= capacity) {
                    items.pop_front();
                    ++dropped;
                }
                items.push_back(std::move(item));
            }
            ready.notify_one();
        }

        std::optional<T> pop(int timeout_ms) {
            std::unique_lock<std::mutex> guard(lock);
            auto has_item_or_closed = [this]() {
                return closed || !items.empty();
            };
            if (timeout_ms < 0) {
                ready.wait(guard, has_item_or_closed);
            }
            else if (!ready.wait_for(guard, std::chrono::milliseconds(timeout_ms), has_item_or_closed)) {
                return std::nullopt;
            }
            if (items.empty()) {
                return std::nullopt;
            }
            auto item = std::move(items.front());
            items.pop_front();
            return item;
        }

        std::optional<T> latest() {
            std::lock_guard<std::mutex> guard(lock);
            if (items.empty()) {
                return std::nullopt;
            }
            auto item = std::move(items.back());
            items.clear();
            return item;
        }

        void clear() {
            std::lock_guard<std::mutex> guard(lock);
            items.clear();
        }

        void close() {
            {
                std::lock_guard<std::mutex> guard(lock);
                closed = true;
                items.clear();
            }
            ready.notify_all();
        }

        void reopen() {
            std::lock_guard<std::mutex> guard(lock);
            closed = false;
            items.clear();
            dropped = 0;
        }

        std::size_t size() const {
            std::lock_guard<std::mutex> guard(lock);
            return items.size();
        }

        std::size_t dropped_count() const {
            std::lock_guard<std::mutex> guard(lock);
            return dropped;
        }
    };

}
